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
    base_url='https://api.mnxcc.com/v1'
)

IMAGE_DIR = 'http://images.cocodataset.org/test2017'

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


E11 = random.choice(example_list_rule1)
E12 = pop_exist(E11, example_list_rule1)
E21 = random.choice(example_list_rule1)
E22 = random.choice(example_list_rule2)
E31 = random.choice(example_list_rule1)
E32 = random.choice(example_list_rule3)
E41 = random.choice(example_list_rule2)
E42 = pop_exist(E41, example_list_rule2)
E51 = random.choice(example_list_rule2)
E52 = random.choice(example_list_rule3)
E61 = random.choice(example_list_rule3)
E62 = pop_exist(E61, example_list_rule3)


example_list = [
    [E11, E12],
    [E21, E22],
    [E31, E32],
    [E41, E42],
    [E51, E52],
    [E61, E62]
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
        print(f"error: {e}")
        return None


def call_gpt4(prompt, image, detail='auto'):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    # zero-shot + few-shot
                    "content": [{"type": "text", "text": prompt[0]}] \
                               + [{"type": "image_url", "image_url": { "url": image[0], "detail": detail}}] \
                               + [{"type": "text", "text": prompt[1]}] \
                               + [{"type": "image_url", "image_url": { "url": image[1], "detail": detail}}] \
                               + [{"type": "text", "text": prompt[2]}] \
                               + [{"type": "image_url", "image_url": { "url": image[2], "detail": detail}}] \
                               + [{"type": "text", "text": prompt[3]}]
                    # text only:
                    # "content": [{"type": "text", "text": question}]
                }
            ],
            max_tokens=500,
            temperature=0.5,
        )
        # print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during answering: {e}")
        return None


def process_jsonl(input_file, output_file):
    count = 0
    right_count = 0

    with open(input_file, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in file:
                data = json.loads(line)
                id = data['id']

                example_2 = example_list[count % 6]

                # 2 - shot
                prompt = ['', '', '', '']
                prompt[0] = f'You are currently a senior expert in spatial relation reasoning. ' \
                         f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                         f'\nGiven the following 2 examples to learn the spatial relation reasoning task:' \
                         f'\nExample1: Input: Image: '
                prompt[1] = f'\n{example_2[0]["text"]} ' \
                         f'\nExample2: Input: Image: '
                prompt[2] = f'\n{example_2[1]["text"]} ' \
                         f'\nInput: Image: '
                # prompt = f'Given the following 3 examples to learn the spatial relation reasoning task:' \
                #          f'\n{example_list[count % 10][0]["text"]} ' \
                #          f'\nYou are currently a senior expert in spatial relation reasoning. ' \
                #          f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                #          f'\nInput: Image: '
                prompt[3] = f'\nQuestion: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '
                
                image_encoded_list = []
                for item in example_2:
                    image_id = item['image']
                    image_url = f'{IMAGE_DIR}/{image_id}'
                    image_content = fetch_image_content(image_url)
                    if image_content is not None:
                        image = Image.open(image_content)
                        image_encoded = encode_image(image)
                        image_encoded_list.append(image_encoded)

                image_id = data['image']
                image_url = f'{IMAGE_DIR}/{image_id}'
                image_content = fetch_image_content(image_url)

                if image_content is not None:
                    image = Image.open(image_content)
                    image_encoded = encode_image(image)
                    image_encoded_list.append(image_encoded)

                    try:
                        model_answer = call_gpt4(prompt, image_encoded_list)
                    except Exception as e:
                        print(e)
                        try:
                            model_answer = call_gpt4(prompt, image_encoded_list)
                        except Exception as e:
                            try:
                                model_answer = call_gpt4(prompt, image_encoded_list)
                            except Exception as e:
                                result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'], "rule": data['rule'], "example": count % 6 + 4}
                                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                                count += 1
                                continue

                    try:
                        # output = model_answer.text.strip().rstrip('.').lower()
                        output = model_answer.strip().rstrip('.').lower()
                    except Exception as e:
                        print(e)
                        output = '--'
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 6 + 4}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1  
                        continue
                    count += 1
                    if output in data['answer'] or data['answer'] in output:
                        result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 6 + 4}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        right_count += 1
                    else:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 6 + 4}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    print(f'{output.lower()}')
                    print(f"{data['answer']}")
                    print(f'right_count: {right_count}')
                    print(f'count: {count}')
                    # print(f'accuracy: {right_count / count}')

            accuracy = right_count / count
            print(accuracy)


input_file_path = "test_en_select_500_sort_cp.jsonl"
output_file_path = "results/gpt4_2_shot_random_new_1.jsonl"

process_jsonl(input_file_path, output_file_path)
