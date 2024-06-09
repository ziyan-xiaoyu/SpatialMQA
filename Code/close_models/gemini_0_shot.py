from time import sleep

import google.generativeai as genai
from io import BytesIO
import requests

genai.configure(api_key="your key",transport="rest")

generation_config = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 480}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

FILE_PATH = 'datasets/test_en_select_500_sort.jsonl'
IMAGE_DIR = 'http://images.cocodataset.org/test2017'
RESULT_FILE_PATH = 'model_results/gemini_0_shot.jsonl'


def fetch_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


import PIL.Image as Image

# few-shot & zero-shot:
model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config, safety_settings=safety_settings)

# text-only:
# model = genai.GenerativeModel('gemini-pro', generation_config=generation_config, safety_settings=safety_settings)

import json

count = 0
right_count = 0
with open(FILE_PATH, 'r', encoding="utf-8") as f, open(RESULT_FILE_PATH, 'a', encoding="utf-8") as fout:
    for line in f:
        sleep(1)
        data = json.loads(line)
        id = data['id']

        # 1 - shot:
        question = f'You are currently a senior expert in spatial relation reasoning. ' \
                   f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                   f'\nInput: Image:<image>, Question: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '

        image_name = data['image']
        image_filepath = f'{IMAGE_DIR}/{image_name}'

        image_content = fetch_image_content(image_filepath)
        if image_content is not None:
            image = Image.open(image_content)
            try:
                response = model.generate_content(
                    [question, image]    # few-shot & zero-shot
                    # [question]         # text-only
                )
            except Exception as e:
                print(e)
                try:
                    response = model.generate_content(
                        [question, image]    # few-shot & zero-shot
                        # [question]         # text-only
                    )
                except Exception as e:
                    try:
                        response = model.generate_content(
                            [question, image]    # few-shot & zero-shot
                            # [question]         # text-only
                        )
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'], "rule": data['rule'], "example": 0}
                        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
            try:
                output = response.text.strip().rstrip('.').lower()
            except Exception as e:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": 0}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1
                continue
            count += 1
            if output in data['answer']:
                result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": 0}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": 0}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            print(f'{output.lower()}')
            print(f"{data['answer']}")
            print(f'right_count: {right_count}')
            print(f'count: {count}')

accuracy = right_count / count
print(accuracy)
