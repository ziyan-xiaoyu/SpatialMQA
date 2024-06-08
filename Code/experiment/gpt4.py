# import openai
from openai import OpenAI
import json
import requests
from io import BytesIO
import PIL.Image as Image
import base64

client = OpenAI(
    api_key='sk-iVKidRNVOP07I3b4Ca428b7354424bDeAbF548B9A3A5A18e',
    base_url='https://api.keya.pw/v1'
)
client.api_base = "https://api.foureast.cn/v1"

IMAGE_DIR = 'http://images.cocodataset.org/test2017'
count = 0
right_count = 0


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
        print(f"There is something wrong when encoding image: {e}")
        return None


def call_gpt4(question: str, image):

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    # "content": prompt_with_image

                    # zero-shot + few-shot
                    "content": [{"type": "text", "text": question}] \
                               + [{"type": "image_url", "image_url": image}]

                    # text only:
                    # "content": [{"type": "text", "text": question}]
                }
            ],
            max_tokens=500,
            # temperature = 0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during answering: {e}")
        return None


def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in file:
                data = json.loads(line)
                id = data['id']
                # question = data['question']

                # text-only
                # question = f'Question: {data["question"]} ' \
                #            f'According to the question, Please use your common sense to choose the answer you think is most likely from the following options seperated by semicolon.' \
                #            f'Note that you only need to answer one word from the options without explaining any reason.' \
                #            f'\nOptions: {"; ".join(data["options"])} '

                # zero-shot
                # question = f'Question: {data["question"]} ' \
                #            f'According to the question, please choose one best option from the following options seperated by semicolon.' \
                #            f'Note that you only need to answer one word from the options without explaining the reason.' \
                #            f'\nOptions: {"; ".join(data["options"])} \nAnswer: '

                # few-shot
                question = f'The following are 4 examples of related questions and corresponding options and answers for spatial relationship recognition:' \
                           f'Sample 1: image: "The picture depicts a clock with the time 12:27. From the perspective of the photographer, the hour hand points to the right of the 12 mark and to the left of the 1 mark." question: "For the clock in the picture, which side of the 1 scale does the hour hand point to?" options: "left; right" answer: "left"' \
                           f'Sample 2: image: "The picture depicts a woman playing tennis. From the perspective of the photographer, the woman is facing us, with the shadow to her left." question: "If you are the lady in the picture, where is your shadow located on you?" options: "front; behind; left; right" answer: "right"' \
                           f'Sample 3: image: "The picture depicts a dining table with cakes, coffee, forks, napkins, etc. From the perspective of the photographer, the napkins are placed on the table and the forks are placed on the napkins to prevent them from getting dirty." question: "Where are the white napkins located on the forks?" options: "on/above; below; front; behind; left; right" answer: "below"' \
                           f'Sample 4: image: "The picture depicts a red warning sign. There are two lines of English words "STOP WAR" written on it. From the perspective of the photographer, the letter T is above the letter W." question: "For the letters on the red warning sign, where is the letter W located on the letter T?" options: "on/above; below; left; right" answer: "below"' \
                           f'Please learn the ideas of spatial relationship recognition from the above samples. Then answer the following question.' \
                           f'Question: {data["question"]} Please choose one best option from the following options seperated by semicolon. Note that you only need to answer one word from the options without explaining any reason. \nOptions: {"; ".join(data["options"])} \nAnswer:'

                image_id = data['image']
                image_url = f'{IMAGE_DIR}/{image_id}'
                image_content = fetch_image_content(image_url)

                if image_content is not None:
                    image = Image.open(image_content)
                    image_encoded = encode_image(image)

                    global count
                    global right_count
                    try:
                        model_answer = call_gpt4(question, image_encoded)
                    except Exception as e:
                        print(e)
                        try:
                            model_answer = call_gpt4(question, image_encoded)
                        except Exception as e:
                            try:
                                model_answer = call_gpt4(question, image_encoded)
                            except Exception as e:
                                result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer']}
                                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                                count += 1
                                continue

                    try:
                        output = model_answer.strip().rstrip('.').lower()
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer']}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
                    count += 1
                    if output == data['answer'] or (output == 'on' and data['answer'] == 'on/above') or (
                            output == 'above' and data['answer'] == 'on/above'):
                        result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer']}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        right_count += 1
                    else:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer']}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    print(f'{output.lower()}')
                    print(f"{data['answer']}")
                    print(f'right_count: {right_count}')
                    print(f'count: {count}')

            accuracy = right_count / count
            print(accuracy)


input_file_path = "../model_exp_results/all_en_test_1076_delete.jsonl"
output_file_path = "../model_exp_results/gpt4_test_1.jsonl"

process_jsonl(input_file_path, output_file_path)
