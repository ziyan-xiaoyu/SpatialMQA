
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

RESULT_FILE_PATH = '/projects/SpatialMQA/output/'
FILE_PATH = '/projects/SpatialMQA/datasets/spatial/'
DEVICE_INDEX = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model_dir = '/projects/Models/'
processor = AutoProcessor.from_pretrained(f"{model_dir}blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(f"{model_dir}blip2-opt-2.7b").to(DEVICE_INDEX)
import json
image_dir = '/projects/SpatialMQA/COCO2017/test2017/'
count = 0
right_count = 0
with open(f'{FILE_PATH}all_en_test_1076_sort.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}blip2_20240531.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = image_dir + image_name
        image = Image.open(image_filepath)
        question = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {"; ".join(options)}. \n Output:'
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE_INDEX)
        predictions = model.generate(**inputs)
        output = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip().rstrip('.')
        count += 1
        if len(output) == 0:
            output = '--'
        if output.lower() in data['answer']:
            result_json = {'id':id,'result':1,"output":output.lower(),"answer":data['answer']}
            fout.write(json.dumps(result_json,ensure_ascii=False)+'\n')
            right_count += 1
        elif data['answer'] in output.lower():
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
