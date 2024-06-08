# adapted from https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/image_captioning.ipynb

# This example demonstrates normal finetuning (w/o peft) - for the sake of keeping the memory
# requirements small it freezes the original pre-trained text and image layers to keep the memory
# requirements to just 40GB. If you have multiple GPUs then you can remove the unfreeze part to
# finetune the whole model. Alternatively use the PEFT solution as shown in
# IDEFICS_finetuning_demo.ipynb notebook which requires only 20GB to finetune the whole model.

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms

device = "cuda:7" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"

# Here we skip some special modules that can't be quantized properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

processor = AutoProcessor.from_pretrained(checkpoint)
# Simply take-off the quantization_config arg if you want to load the original model
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="cuda:0")


def check_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids,
                                   max_new_tokens=max_new_tokens, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
prompts = [
    # "Instruction: provide an answer to the question. Use the image to answer.\n",
    url,
    "Question: What's on the picture? Answer:",
]
check_inference(model, processor, prompts)

train_ds = load_dataset("json", data_files='/projects/SpatialMQA/datasets/idefics_train/train_3780.jsonl')['train']
eval_ds = load_dataset("json", data_files='/projects/SpatialMQA/datasets/idefics_train/dev_536.jsonl')['train']


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    image = Image.open(image)
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch)):
        prompts.append(
            [
                example_batch["image"][i],
                example_batch["text"][i],
            ],
        )

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]

    return inputs


train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)

model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)

num_train_epochs = 10

training_args = TrainingArguments(
    output_dir=f"/projects/SpatialMQA/finetune_models/models_arg/idefics_lora_20240521",
    num_train_epochs=5,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,
    save_total_limit=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to=None,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

model.save_pretrained("/projects/SpatialMQA/finetune_models/models_arg/idefics_lora_20240521")

check_inference(model, processor, prompts, max_new_tokens=100)
