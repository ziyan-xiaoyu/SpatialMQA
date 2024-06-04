# 从图像数据集中复制图片
import json
import os
import shutil


def copy_images(jsonl_file, source_dir, destination_dir):
    # 创建目标文件夹
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    count = 0
    # 打开jsonl文件
    with open(jsonl_file, 'r', encoding='gbk', errors='ignore') as f:
        # 逐行读取json数据
        for line in f:
            count += 1
            data = json.loads(line)
            image_filename = data['image']

            # 构建源文件路径
            source_path = os.path.join(source_dir, image_filename)

            # 如果源文件存在，则进行复制操作
            if os.path.exists(source_path):
                destination_path = os.path.join(destination_dir, image_filename)
                shutil.copyfile(source_path, destination_path)
            else:
                print(f"Source file {image_filename} not found in {source_dir}")

    print(jsonl_file+': '+str(count))
    print("图片复制完成")


# 使用示例
jsonl_file = os.listdir('dataset/SpatialMQA_shuffled/')
for i in range(len(jsonl_file)):
    jsonl_file[i] = 'dataset/SpatialMQA_shuffled/'+jsonl_file[i]
source_dir = 'E:\\0实验室\\项目\\GPT误导性研究\\空间推理能力研究\\相关数据集\\coco\\test2017'
destination_dir = 'dataset/images'

for file in jsonl_file:
    copy_images(file, source_dir, destination_dir)
