# import openai
from openai import OpenAI
import json
import requests
from io import BytesIO
import PIL.Image as Image
import base64


client = OpenAI(
    api_key='your key',
    base_url='https://gpt.mnxcc.com/v1'
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
        print(f"encoding error: {e}")
        return None


def call_gpt4(prompt: str, image):
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    # zero-shot + few-shot
                    "content": [{"type": "text", "text": prompt}] \
                               + [{"type": "image_url", "image_url": image}]
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


def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in file:
                data = json.loads(line)
                id = data['id']

                # zero - shot
                prompt = f'You are currently a senior expert in spatial relation reasoning. ' \
                         f'\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.' \
                         f'\nInput: Image:<image>, Question: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput: '

                image_id = data['image']
                image_url = f'{IMAGE_DIR}/{image_id}'
                image_content = fetch_image_content(image_url)

                if image_content is not None:
                    image = Image.open(image_content)
                    image_encoded = encode_image(image)

                    global count
                    global right_count
                    try:
                        model_answer = call_gpt4(prompt, image_encoded)
                    except Exception as e:
                        print(e)
                        try:
                            model_answer = call_gpt4(prompt, image_encoded)
                        except Exception as e:
                            try:
                                model_answer = call_gpt4(prompt, image_encoded)
                            except Exception as e:
                                result_json = {"id": id, "result": 0, "output": "--", "answer": data['answer'],
                                               "rule": data['rule'], "example": 0}
                                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                                count += 1
                                continue

                    try:
                        # output = model_answer.text.strip().rstrip('.').lower()
                        output = model_answer.strip().rstrip('.').lower()
                    except Exception as e:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": 0}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        count += 1
                        continue
                    count += 1
                    if output in data['answer']:
                        result_json = {"id": id, "result": 1, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": 0}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                        right_count += 1
                    else:
                        result_json = {"id": id, "result": 0, "output": output.lower(), "answer": data['answer'],
                                       "rule": data['rule'], "example": 0}
                        out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    print(f'{output.lower()}')
                    print(f"{data['answer']}")
                    print(f'right_count: {right_count}')
                    print(f'count: {count}')
                    # print(f'accuracy: {right_count / count}')

            accuracy = right_count / count
            print(accuracy)


input_file_path = "datasets/test_en_500.jsonl"
output_file_path = "model_results/gpt4_zero_shot.jsonl"


process_jsonl(input_file_path, output_file_path)
