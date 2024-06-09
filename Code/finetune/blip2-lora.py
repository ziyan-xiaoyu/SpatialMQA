import os

import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import Blip2Processor, AutoProcessor, Blip2ForConditionalGeneration, \
    AutoModelForVisualQuestionAnswering
from datasets import load_dataset
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from peft import LoraConfig, get_peft_model

model_dir = '/projects/Models/'
processor = Blip2Processor.from_pretrained(f"{model_dir}blip2-opt-2.7b")
image_dir = '/projects/SpatialMQA/COCO2017/test2017/'

train_ds = load_dataset("json", data_files="/projects/SpatialMQA/datasets/blip2_train/train_3780.jsonl",
                        split="train[:100%]")
eval_ds = load_dataset("json", data_files="/projects/SpatialMQA/datasets/blip2_train/dev_536.jsonl",
                       split="train[:100%]")
print("Training sets: {} - Validating set: {}".format(len(train_ds), len(eval_ds)))

device = "cuda:7" if torch.cuda.is_available() else "cpu"

model = Blip2ForConditionalGeneration.from_pretrained(f"{model_dir}blip2-opt-2.7b", device_map="cuda:7",
                                                      load_in_8bit=True)

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config).to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


class ImageCaptioningDataset(Dataset):
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
        image_path = f"/projects/SpatialMQA/COCO2017/test2017/{image_id}"
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(images=image, text=question, return_tensors="pt")

        labels = self.processor.tokenizer.tokenize(answer, return_tensors='pt')
        labels = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(labels)).unsqueeze(0)
        encoding["labels"] = torch.cat((labels, torch.tensor([50118]).unsqueeze(0)), dim=1)

        # remove batch dimension
        for k, v in encoding.items():  encoding[k] = v.squeeze()
        return encoding


train_dataset = ImageCaptioningDataset(dataset=train_ds, processor=processor)
valid_dataset = ImageCaptioningDataset(dataset=eval_ds, processor=processor)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

cal_num = 2
num_epochs = 30
patience = 5
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss(ignore_index=1)

for epoch in range(num_epochs):
    epoch_loss = 0
    cal_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask).logits

        loss = criterion(outputs.view(-1, outputs.shape[-1])[:labels.shape[1], :].contiguous(),
                         labels.view(-1).contiguous())
        epoch_loss += loss.item()
        optimizer.zero_grad()

        cal_loss += loss
        if (idx + 1) % cal_num == 0 or idx == len(train_dataloader) - 1:
            if (idx + 1) % cal_num == 0:
                cal_loss = cal_loss / cal_num
            else:
                cal_loss = cal_loss / ((idx + 1) % cal_num)
            print('loss:', cal_loss)
            scaler.scale(cal_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cal_loss = 0

    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        # print('labels:',labels)
        # outputs = model(input_ids=input_ids,
        #             pixel_values=pixel_values,
        #             attention_mask=attention_mask,
        #             labels=labels)
        # loss = outputs.loss

        # labels_mask = batch.pop('labels_mask').to(device)
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask).logits
        # loss = criterion(outputs.view(-1, outputs.shape[-1])[:8, :].contiguous() * labels_mask.view(-1, 1).contiguous(), labels.view(-1).contiguous())
        loss = criterion(outputs.view(-1, outputs.shape[-1])[:labels.shape[1], :].contiguous(),
                         labels.view(-1).contiguous())

        eval_loss += loss.item()
        # break

    # break

    tracking_information.append(
        (epoch_loss / len(train_dataloader), eval_loss / len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_dataloader),
                                                                          eval_loss / len(valid_dataloader),
                                                                          optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained(f"/projects/SpatialMQA/finetune_models/models_arg/blip2_lora_20240531/epoch-{epoch}",
                              from_pt=True)
        print(f"Saved model to /projects/SpatialMQA/finetune_models/models_arg/blip2_lora_20240531")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")
