import json
import os
import shutil


def copy_images(jsonl_file, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    count = 0
    with open(jsonl_file, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            count += 1
            data = json.loads(line)
            image_filename = data['image']

            source_path = os.path.join(source_dir, image_filename)

            if os.path.exists(source_path):
                destination_path = os.path.join(destination_dir, image_filename)
                shutil.copyfile(source_path, destination_path)
            else:
                print(f"Source file {image_filename} not found in {source_dir}")

    print(jsonl_file+': '+str(count))
    print("Copy successful!")


jsonl_file = os.listdir('../dataset/')
for i in range(len(jsonl_file)):
    jsonl_file[i] = '../dataset/'+jsonl_file[i]
source_dir = '../COCO2017'
destination_dir = '../relevant_images'

for file in jsonl_file:
    copy_images(file, source_dir, destination_dir)
