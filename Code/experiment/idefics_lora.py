# this is a demo of inference of IDEFICS-9B which needs about 20GB of GPU memory

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

device = "cuda:7" if torch.cuda.is_available() else "cpu"
RESULT_FILE_PATH = '/projects/SpatialMQA/output/idefics-10/'
FILE_PATH = '/projects/SpatialMQA/datasets/spatial/'

lora_point = '/projects/SpatialMQA/finetune_models/models_arg/idefics_lora_20240521/checkpoint-35'

from peft import PeftModel, PeftConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

config = PeftConfig.from_pretrained(lora_point)
processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)
model = IdeficsForVisionText2Text.from_pretrained(config.base_model_name_or_path,quantization_config=bnb_config,device_map="cuda:7")
model = PeftModel.from_pretrained(model, lora_point).to(device)


import json
import re
from PIL import Image
image_dir = '/projects/SpatialMQA/COCO2017/test2017/'
count = 0
right_count = 0

def check_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=max_new_tokens, early_stopping=True,
                                   do_sample=True, temperature=0.7)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


with open(f'{FILE_PATH}all_en_test_1076_sort.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}idefics_lora_7.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = image_dir + image_name
        image = Image.open(image_filepath)
        question = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {"; ".join(options)}. \nOutput:'
        prompts = [
            image,
            question,
        ]

        generated_text = check_inference(model, processor, prompts, max_new_tokens=5)
        print(f'generated_text:\n{generated_text}')
        output = generated_text.lower().split('answer:')[-1].split('\n')[0].strip().rstrip('.')
        count += 1
        if len(output) == 0:
            output = '--'
        if output.lower() in data['answer']:
            result_json = {'id':id,'result':1,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
            right_count += 1
        elif data['answer'] in output:
            result_json = {'id':id,'result':1,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
            right_count += 1
        else:
            result_json = {'id':id,'result':0,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
        print(f'{output.lower()}')
        print(f"{data['answer']}")
        print(f'right_count: {right_count}')
        print(f'count: {count}')
        print(f'accuracy: {right_count/count}')

accuracy = right_count/count
print(f'accuracy: {accuracy}')
