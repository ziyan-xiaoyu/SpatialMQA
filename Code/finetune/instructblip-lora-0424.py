import os

import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

model_dir = '/projects/Models/'
image_dir = '/projects/SpatialMQA/COCO2017/test2017/'

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q", "k"]
)

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

device = "cuda:6" if torch.cuda.is_available() else "cpu"

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")

model = get_peft_model(model, config).to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        question = self.dataset[idx]['question']
        answer = str(self.dataset[idx]['answer'])
        image_id = self.dataset[idx]['image']
        image_path = f"{image_dir}{image_id}"
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(images=image, text=question, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, pad_to_max_length=True, return_tensors='pt'
        )

        encoding["labels"] = labels
        # remove batch dimension
        for k, v in encoding.items():  encoding[k] = v.squeeze()
        print(encoding['labels'], answer)
        return encoding


training_dataset = load_dataset("json", data_files="/projects/SpatialMQA/datasets/instructblip_train/train_3780.jsonl",
                                split="train[:100%]")
valid_dataset = load_dataset("json", data_files="/projects/SpatialMQA/datasets/instructblip_train/dev_536.jsonl",
                             split="train[:100%]")
print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))

train_dataset = VQADataset(dataset=training_dataset,
                           processor=processor)
valid_dataset = VQADataset(dataset=valid_dataset,
                           processor=processor)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 30
patience = 5
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        qformer_input_ids = batch.pop('qformer_input_ids').to(device)
        qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids,
                            qformer_input_ids=qformer_input_ids,
                            qformer_attention_mask=qformer_attention_mask,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels
                            )
        loss = outputs.loss
        print('loss:', loss)
        epoch_loss += loss.item()
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        qformer_input_ids = batch.pop('qformer_input_ids').to(device)
        qformer_attention_mask = batch.pop('qformer_attention_mask').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        outputs = model(input_ids=input_ids,
                        qformer_input_ids=qformer_input_ids,
                        qformer_attention_mask=qformer_attention_mask,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        eval_loss += loss.item()

    tracking_information.append(
        (epoch_loss / len(train_dataloader), eval_loss / len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_dataloader),
                                                                          eval_loss / len(valid_dataloader),
                                                                          optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("/projects/SpatialMQA/finetune_models/models_arg/instructblip_lora_20240530",
                              from_pt=True)
        print("Saved model to /projects/SpatialMQA/finetune_models/models_arg/instructblip_lora_20240530")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")
