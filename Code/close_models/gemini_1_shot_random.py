from time import sleep
import PIL.Image as Image
import google.generativeai as genai
from io import BytesIO
import requests
import random
import json

genai.configure(api_key="your key",transport="rest")

generation_config = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 480}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

FILE_PATH = './datasets/test_en_select_500_sort1.jsonl'
IMAGE_DIR = 'http://images.cocodataset.org/test2017'
RESULT_FILE_PATH = './model_results/gemini_1_shot_random.jsonl'


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


example_list = [
    random.choice(example_list_rule1),
    random.choice(example_list_rule2),
    random.choice(example_list_rule3)
]


def fetch_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


def choose_1_example():
    example_1 = random.choice(example_list_rule1 + example_list_rule1 + example_list_rule1)
    return example_1


# few-shot & zero-shot:
model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config, safety_settings=safety_settings)

# text-only:
# model = genai.GenerativeModel('gemini-pro', generation_config=generation_config, safety_settings=safety_settings)


count = 0
right_count = 0
with open(FILE_PATH, 'r', encoding="utf-8") as f, open(RESULT_FILE_PATH, 'a', encoding="utf-8") as fout:
    for line in f:
        sleep(1)
        data = json.loads(line)
        id = data['id']

        example_1 = choose_1_example()

        # 1 - shot:
        # question = f'You are currently a senior expert in spatial relation reasoning. ' \
        #            f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
        #            f'\nGiven the following 1 examples to learn the spatial relation reasoning task:' \
        #            f'\n{example_1} ' \
        #            f'Input: Image:<image>, Question: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '
        prompt1 = f'You are currently a senior expert in spatial relation reasoning. ' \
                  f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                  f'\nGiven the following 1 examples to learn the spatial relation reasoning task:' \
                  f'\nInput: Image: '
        prompt2 = f'\n{example_1["text"]} ' \
                  f'\nInput: Image: '
        prompt3 = f'\nQuestion: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '

        image_eg1_id = example_1["image"]
        image_eg1_url = f'{IMAGE_DIR}/{image_eg1_id}'
        image_eg1_content = fetch_image_content(image_eg1_url)

        image_id = data['image']
        image_url = f'{IMAGE_DIR}/{image_id}'
        image_content = fetch_image_content(image_url)

        if (image_eg1_content is not None) and (image_content is not None):
            image_eg1 = Image.open(image_eg1_content)
            image = Image.open(image_content)
            try:
                response = model.generate_content(
                    [prompt1, image_eg1, prompt2, image, prompt3]
                )
            except Exception as e:
                print(e)
                try:
                    response = model.generate_content(
                        [prompt1, image_eg1, prompt2, image, prompt3]
                    )  
                except Exception as e:
                    try:
                        response = model.generate_content(
                            [prompt1, image_eg1, prompt2, image, prompt3]
                        )  
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'], "rule": data['rule'], "example": count % 3 + 1}
                        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1  
                        continue
            try:
                output = response.text.strip().rstrip('.').lower()
            except Exception as e:
                print(e)
                output = '--'
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 3 + 1}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1  
                continue
            count += 1
            if output in data['answer'] or data['answer'] in output:
                result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 3 + 1}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'], "rule": data['rule'], "example": count % 3 + 1}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            print(f'{output.lower()}')
            print(f"{data['answer']}")
            print(f'right_count: {right_count}')
            print(f'count: {count}')
            # print(f'accuracy: {right_count / count}')

accuracy = right_count / count
print(accuracy)
