import google.generativeai as genai
from io import BytesIO
import requests
import PIL.Image as Image

genai.configure(api_key="AIzaSyDhPawyaVgt8JwT2kn-q3N2LADoe7x2O98", transport="rest")

generation_config = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 480}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

FILE_PATH = 'all_en_test_1076_sort.jsonl'
IMAGE_DIR = 'http://images.cocodataset.org/test2017'
RESULT_FILE_PATH = 'gemini_test_only_1.jsonl'


def fetch_image_content(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


# few-shot & zero-shot:
# model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config, safety_settings=safety_settings)

# text-only:
model = genai.GenerativeModel('gemini-pro', generation_config=generation_config, safety_settings=safety_settings)

import json

count = 0
right_count = 0
with open(FILE_PATH, 'r', encoding="utf-8") as f, open(RESULT_FILE_PATH, 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        id = data['id']
        # text-only:
        question = f'Question: {data["question"]} ' \
                   f'According to the question, Please use your common sense to choose the answer you think is most likely from the following options seperated by semicolon.' \
                   f'Note that you only need to answer one word from the options without explaining any reason.' \
                   f'\nOptions: {"; ".join(data["options"])} '

        # zero-shot:
        # question = f'Question: {data["question"]} ' \
        #            f'According to the question, please choose one best option from the following options seperated by semicolon.' \
        #            f'Note that you only need to answer one word from the options without explaining any reason.' \
        #            f'\nOptions: {"; ".join(data["options"])} \nAnswer:'

        # few-shot:
        # question = f'The following are 4 examples of related questions and corresponding options and answers for spatial relationship recognition:' \
        #            f'Sample 1: image: "The picture depicts a clock with the time 12:27. From the perspective of the photographer, the hour hand points to the right of the 12 mark and to the left of the 1 mark." question: "For the clock in the picture, which side of the 1 scale does the hour hand point to?" options: "left; right" answer: "left"' \
        #            f'Sample 2: image: "The picture depicts a woman playing tennis. From the perspective of the photographer, the woman is facing us, with the shadow to her left." question: "If you are the lady in the picture, where is your shadow located on you?" options: "front; behind; left; right" answer: "right"' \
        #            f'Sample 3: image: "The picture depicts a dining table with cakes, coffee, forks, napkins, etc. From the perspective of the photographer, the napkins are placed on the table and the forks are placed on the napkins to prevent them from getting dirty." question: "Where are the white napkins located on the forks?" options: "on/above; below; front; behind; left; right" answer: "below"' \
        #            f'Sample 4: image: "The picture depicts a red warning sign. There are two lines of English words "STOP WAR" written on it. From the perspective of the photographer, the letter T is above the letter W." question: "For the letters on the red warning sign, where is the letter W located on the letter T?" options: "on/above; below; left; right" answer: "below"' \
        #            f'Please learn the ideas of spatial relationship recognition from the above samples. Then answer the following question.' \
        #            f'Question: {data["question"]} Please choose one best option from the following options seperated by semicolon. Note that you only need to answer one word from the options without explaining any reason. \nOptions: {"; ".join(data["options"])} \nAnswer:'

        image_name = data['image']
        image_filepath = f'{IMAGE_DIR}/{image_name}'

        image_content = fetch_image_content(image_filepath)
        if image_content is not None:
            image = Image.open(image_content)

            # response = model.generate_content(
            #     [
            #         question,
            #         image
            #     ],
            #     stream=True
            # )
            # response.resolve()
            try:
                response = model.generate_content(
                    # [question, image]    # few-shot & zero-shot
                    [question]         # text-only
                )
            except Exception as e:
                print(e)
                try:
                    response = model.generate_content(
                        # [question, image]    # few-shot & zero-shot
                        [question]         # text-only
                    )
                except Exception as e:
                    try:
                        response = model.generate_content(
                            # [question, image]    # few-shot & zero-shot
                            [question]         # text-only
                        )
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer']}
                        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
            try:
                output = response.text.strip().rstrip('.').lower()
            except Exception as e:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1
                continue
            count += 1
            if output == data['answer'] or (output == 'on' and data['answer'] == 'on/above') or (
                            output == 'above' and data['answer'] == 'on/above'):
                result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            print(f'{output.lower()}')
            print(f"{data['answer']}")
            print(f'right_count: {right_count}')
            print(f'count: {count}')

accuracy = right_count / count
print(accuracy)
