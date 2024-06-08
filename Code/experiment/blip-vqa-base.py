import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import json

RESULT_FILE_PATH = '/projects/SpatialMQA/output/'
FILE_PATH = '/projects/SpatialMQA/datasets/spatial/'
DEVICE_INDEX = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(f"Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(f"Salesforce/blip-vqa-base").to(DEVICE_INDEX)

image_dir = '/projects/SpatialMQA/COCO2017/test2017/'
count = 0
right_count = 0
with open(f'{FILE_PATH}all_en_test_1076_sort.jsonl', 'r', encoding="utf-8") as f,open(f'{RESULT_FILE_PATH}blip.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = image_dir + image_name
        image = Image.open(image_filepath).convert('RGB')
        question = f'{question} {",".join(options[:-1])} or {options[-1]}'
        print(question)
        inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE_INDEX)
        predictions = model.generate(**inputs)
        output = (processor.decode(predictions[0], skip_special_tokens=True))
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

