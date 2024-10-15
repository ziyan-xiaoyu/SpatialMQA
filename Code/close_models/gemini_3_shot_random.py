from time import sleep

import google.generativeai as genai
from io import BytesIO
import requests
import random

genai.configure(api_key="your key", transport="rest")

generation_config = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 480}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

FILE_PATH = 'test_en_select_500_sort_cp_2.jsonl'
IMAGE_DIR = 'http://images.cocodataset.org/test2017'
RESULT_FILE_PATH = 'results/gemini_3_shot_random_cp.jsonl'


example_list_rule1 = [
    {"id": 1, "image": "000000358641.jpg", "text": "Question: For the clock in the picture, which side of the 1 scale does the hour hand point to?, Options: left of; right of. \nOutput: right of."},
    {"id": 2, "image": "000000209618.jpg", "text": "Question: Where is the white plate located relative to the glass?, Options: in front of; behind; left of; right of. \nOutput: in front of."},
    {"id": 3, "image": "000000010682.jpg", "text": "Question: For the letters on the warning sign, where is the letter W located relative to the letter O?, Options: on/above; below; left of; right of. \nOutput: below."}
]

example_list_rule2 = [
    {"id": 1, "image": "000000057664.jpg", "text": "Question: If you are the person skiing in the picture, where is your shadow located relative to you?, Options: in front of; behind; left of; right of. \nOutput: right of."},
    {"id": 2, "image": "000000073924.jpg", "text": "Question: If you were a player playing on the court, where would the tennis ball be located relative to you?, Options: on/above; below; in front of; behind; left of; right of. \nOutput: on/above."},
    {"id": 3, "image": "000000022707.jpg", "text": "Question: If you were the little girl in the picture, where would the window be located relative to you?, Options: in front of; behind; left of; right of. \nOutput: behind."}
]

example_list_rule3 = [
    {"id": 1, "image": "000000139664.jpg", "text": "Question: If you are the driver of the bus in the picture, from your perspective, where is the stroller located relative to the bus?, Options: in front of; behind; left of; right of. \nOutput: left of."},
    {"id": 2, "image": "000000221101.jpg", "text": "If you are sitting in front of the computer in the picture, where is the scissors located relative to the laptop from your perspective?, Options: on/above; below; in front of; behind; left of; right of. \nOutput: right of."},
    {"id": 3, "image": "000000164692.jpg", "text": "Question: If you are the driver of the white car in the picture, from your perspective, where is the motorcycle located relative to the car?, Options: in front of; behind; left of; right of. \nOutput: behind."}
]


def pop_exist(data, datalist):
    data_list = datalist.copy()
    data_list.remove(data)
    return random.choice(data_list)


E11 = example_list_rule1[0]
E12 = example_list_rule1[1]
E13 = example_list_rule1[2]
E21 = random.choice(example_list_rule1)
E22 = pop_exist(E21, example_list_rule1)
E23 = random.choice(example_list_rule2)
E31 = random.choice(example_list_rule1)
E32 = pop_exist(E31, example_list_rule1)
E33 = random.choice(example_list_rule3)
E41 = random.choice(example_list_rule1)
E42 = random.choice(example_list_rule2)
E43 = pop_exist(E42, example_list_rule2)
E51 = random.choice(example_list_rule1)
E52 = random.choice(example_list_rule2)
E53 = random.choice(example_list_rule3)
E61 = random.choice(example_list_rule1)
E62 = random.choice(example_list_rule3)
E63 = pop_exist(E62, example_list_rule3)
E71 = example_list_rule2[0]
E72 = example_list_rule2[1]
E73 = example_list_rule2[2]
E81 = random.choice(example_list_rule2)
E82 = pop_exist(E81, example_list_rule2)
E83 = random.choice(example_list_rule3)
E91 = random.choice(example_list_rule2)
E92 = random.choice(example_list_rule3)
E93 = pop_exist(E92, example_list_rule3)
E101 = example_list_rule3[0]
E102 = example_list_rule3[1]
E103 = example_list_rule3[2]


example_list = [
    [E11, E12, E13],
    [E21, E22, E23],
    [E31, E32, E33],
    [E41, E42, E43],
    [E51, E52, E53],
    [E61, E62, E63],
    [E71, E72, E73],
    [E81, E82, E83],
    [E91, E92, E93],
    [E101, E102, E103]
]


def fetch_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


import PIL.Image as Image

# few-shot & zero-shot:
model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config, safety_settings=safety_settings)

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

        example_3 = example_list[count % 10]

        # 3 - shot
        prompt = ['', '', '', '', '']
        prompt[0] = f'You are currently a senior expert in spatial relation reasoning. ' \
                    f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                    f'\nGiven the following 3 examples to learn the spatial relation reasoning task:' \
                    f'\nExample1: Input: Image: '
        prompt[1] = f'\n{example_3[0]["text"]} ' \
                    f'\nExample2: Input: Image: '
        prompt[2] = f'\n{example_3[1]["text"]} ' \
                    f'\nExample3: Input: Image: '
        prompt[3] = f'\n{example_3[2]["text"]} ' \
                    f'\nInput: Image: '

        # prompt = f'Given the following 3 examples to learn the spatial relation reasoning task:' \
        #          f'\n{example_list[count % 10][0]["text"]} ' \
        #          f'\nYou are currently a senior expert in spatial relation reasoning. ' \
        #          f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
        #          f'\nInput: Image: '
        prompt[4] = f'\nQuestion: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '

        image_list = []
        for item in example_3:
            image_id = item['image']
            image_url = f'{IMAGE_DIR}/{image_id}'
            image_content = fetch_image_content(image_url)
            if image_content is not None:
                image = Image.open(image_content)
                image_list.append(image)

        image_id = data['image']
        image_url = f'{IMAGE_DIR}/{image_id}'
        image_content = fetch_image_content(image_url)

        if image_content is not None:
            image = Image.open(image_content)
            image_list.append(image)
            try:
                response = model.generate_content(
                    [prompt[0], image_list[0], prompt[1], image_list[1], prompt[2], image_list[2], prompt[3], image_list[3], prompt[4]]    # few-shot & zero-shot
                    # [question]         # text-only
                )
            except Exception as e:
                print(e)
                try:
                    response = model.generate_content(
                        [prompt[0], image_list[0], prompt[1], image_list[1], prompt[2], image_list[2], prompt[3], image_list[3], prompt[4]]    # few-shot & zero-shot
                        # [question]         # text-only
                    )
                except Exception as e:
                    try:
                        response = model.generate_content(
                            [prompt[0], image_list[0], prompt[1], image_list[1], prompt[2], image_list[2], prompt[3], image_list[3], prompt[4]]    # few-shot & zero-shot
                            # [question]         # text-only
                        )
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'], "rule": data['rule'], "example": count % 10 + 10}
                        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
            try:
                output = response.text.strip().rstrip('.').lower()
            except Exception as e:
                print(e)
                output = '--'
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 10 + 10}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1
                continue
            count += 1
            if output in data['answer'] or data['answer'] in output:
                result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 10 + 10}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 10 + 10}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            print(f'{output.lower()}')
            print(f"{data['answer']}")
            print(f'right_count: {right_count}')
            print(f'count: {count}')

accuracy = right_count / count
print(accuracy)
