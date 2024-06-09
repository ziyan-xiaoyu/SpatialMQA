# import openai
from openai import OpenAI
import json
import requests
from io import BytesIO
import PIL.Image as Image
import base64
import random


client = OpenAI(
    api_key='your key',
    base_url='https://gpt.mnxcc.com/v1'
)
client.api_base = "https://api.foureast.cn/v1"

IMAGE_DIR = 'http://images.cocodataset.org/test2017'

example_list_rule1 = [
    'Input: Image: "The picture depicts a clock with the time 12:27. From the perspective of the photographer, the clock is facing you and the 12 scale is directly above the 6 scale.", Question: "For the clock in the picture, which side of the 1 scale does the hour hand point to?", Options: "left of; right of". \nOutput: left of.',
    'Input: Image: "The picture depicts a dining table with white plate, glass, etc. From the perspective of the photographer, the glass cup is further away from you than the white plate.", Question: "Where is the white plate located relative to the glass?", Options: "in front of, behind, left of, right of". \nOutput: in front of.',
    'Input: Image: "The picture depicts a red warning sign with two lines of English words "STOP WAR" written on it. From the perspective of the photographer, the letter O is above the letter A." Question: "For the letters on the red warning sign, where is the letter W located on the letter T?", Options: "on/above; below; left of; right of". \nOutput: below.'
]

example_list_rule2 = [
    'Input: Image: "The picture depicts a man playing baseball. From the perspective of the photographer, the man is facing you, with the shadow to his left.", Question: "If you are the man in the picture, where is your shadow located relative to you?", Options: "in front of; behind; left of; right of". \nOutput: right of.',
    'Input: Image: "The picture depicts a woman playing tennis. From the perspective of the photographer, the tennis ball happened to fly over the head of woman", Question: "If you are the woman in the picture, where is the tennis ball located relative to you?", Options: "on/above; below; in front of; behind; left of; right of". \nOutput: on/above.',
    'Input: Image: "The picture depicts a child sitting on the bed. From the perspective of the photographer, the child is facing you and the window is located on the wall behind the bed.", Question: "If you are the child in the picture, where is the window located relative to you?", Options: "in front of; behind; left of; right of". \nOutput: behind.'
]

example_list_rule3 = [
    'Input: Image: "This picture depicts a bus and a stroller. From the perspective of the photographer, the bus is heading towards you, and the stroller is located on the right side of the bus.", Question: "If you are the driver of the bus in the picture, from your perspective, where is the stroller located on the bus?", Options: "in front of; behind; left of; right of". \nOutput: left of.',
    'Input: Image: "The picture depicts a dask table with laptop, scissors, etc. From the perspective of the photographer, the laptop is facing to the left front, and there is a stack of books on the right front of the laptop with a pair of scissors on them.", Question: "If you are sitting in front of the computer in the picture, where is the scissors located relative to the laptop from your perspective?", Options: "on/above; below; in front of; behind; left of; right of". \nOutput: right of.',
    'Input: Image: "The picture depicts a white car and a motorcycle. From the perspective of the photographer, there is a motorcycle in the middle, with a white car behind it and the rear of car facing you.", Question: "If you are the driver of the white car in the picture, from your perspective, where is the motorcycle located relative to the car?", Options: "in front of; behind; left of; right of". \nOutput: behind.'
]


def pop_exist(data, datalist):
    data_list = datalist.copy()
    data_list.remove(data)
    # print(len(datalist))
    # print(len(data_list))
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
    'Example 1 ' + E11 + '\nExample 2 ' + E12 + '\nExample 3 ' + E13,
    'Example 1 ' + E21 + '\nExample 2 ' + E22 + '\nExample 3 ' + E23,
    'Example 1 ' + E31 + '\nExample 2 ' + E32 + '\nExample 3 ' + E33,
    'Example 1 ' + E41 + '\nExample 2 ' + E42 + '\nExample 3 ' + E43,
    'Example 1 ' + E51 + '\nExample 2 ' + E52 + '\nExample 3 ' + E53,
    'Example 1 ' + E61 + '\nExample 2 ' + E62 + '\nExample 3 ' + E63,
    'Example 1 ' + E71 + '\nExample 2 ' + E72 + '\nExample 3 ' + E73,
    'Example 1 ' + E81 + '\nExample 2 ' + E82 + '\nExample 3 ' + E83,
    'Example 1 ' + E91 + '\nExample 2 ' + E92 + '\nExample 3 ' + E93,
    'Example 1 ' + E101 + '\nExample 2 ' + E102 + '\nExample 3 ' + E103
]

# count = 0
# right_count = 0


def fetch_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


def encode_image(image):
    if image is None:
        return None  

    buffered = BytesIO()
    try:
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return 'data:image/jpeg;base64,' + img_str
    except Exception as e:
        print(f"encoding error: {e}")
        return None


def call_gpt4(prompt: str, question: str, image):
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    # zero-shot + few-shot
                    "content": [{"type": "text", "text": prompt}] \
                               + [{"type": "image_url", "image_url": image}] \
                               + [{"type": "text", "text": question}]

                    # text only:
                    # "content": [{"type": "text", "text": question}]
                }
            ],
            max_tokens=500,
            temperature=0.3,
        )
        # print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during answering: {e}")
        return None


# 读取 JSONL 文件
def process_jsonl(input_file, output_file):
    count = 0
    right_count = 0

    with open(input_file, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in file:
                data = json.loads(line)
                id = data['id']

                # 3 - shot
                prompt = f'You are currently a senior expert in spatial relation reasoning. ' \
                       f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                       f'\nGiven the following 3 examples to learn the spatial relation reasoning task:' \
                       f'\n{example_list[count % 10]} ' \
                       f'\nInput: Image: '

                question = f'\nQuestion: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '

                image_id = data['image']
                image_url = f'{IMAGE_DIR}/{image_id}'
                image_content = fetch_image_content(image_url)

                if image_content is not None:
                    image = Image.open(image_content)
                    image_encoded = encode_image(image)

                    try:
                        model_answer = call_gpt4(prompt, question, image_encoded)
                    except Exception as e:
                        print(e)
                        try:
                            model_answer = call_gpt4(prompt, question, image_encoded)
                        except Exception as e:
                            try:
                                model_answer = call_gpt4(prompt, question, image_encoded)
                            except Exception as e:
                                result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'],
                                               "rule": data['rule'], "example": count % 10 + 10}
                                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                                count += 1
                                continue

                    try:
                        # output = model_answer.text.strip().rstrip('.').lower()
                        output = model_answer.strip().rstrip('.').lower()
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": count % 10 + 10}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
                    count += 1
                    if output in data['answer']:
                        result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": count % 10 + 10}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        right_count += 1
                    else:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": count % 10 + 10}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    print(f'{output.lower()}')
                    print(f"{data['answer']}")
                    print(f'right_count: {right_count}')
                    print(f'count: {count}')
                    # print(f'accuracy: {right_count / count}')

            accuracy = right_count / count
            print(accuracy)


input_file_path = "datasets/test_en_500.jsonl"
output_file_path = "model_results/gpt4_3_shot.jsonl"


process_jsonl(input_file_path, output_file_path)
