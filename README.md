<br />
<p align="center">
  <h1 align="center"> 🔭 Can Multimodal Large Language Models Understand Spatial Relations</h1>
  <h3 align="center">SpatialMQA: A new dataset for spatial reasoning of MLLMs.</h3>
  
  <p align="center">  
<!--     <a href="https://arxiv.org/abs/2205.00363">arxiv</a> -->
    ·
    <a href="https://github.com/ziyan-xiaoyu/SpatialMQA/blob/master/Dataset/dataset/train.jsonl">dataset</a>
    ·
    <a href="https://github.com/ziyan-xiaoyu/SpatialMQA/blob/master/LICENSE">license</a>
<!--     <a href="https://paperswithcode.com/sota/visual-reasoning-on-vsr">benchmark</a> -->
    
  </p>
</p>


## Contents

- [SpatialMQA](#Contents)
  - [Overview](#1-Overview)
    - [Examples](#Examples)
    - [Detail Information](#Detail-Information)
  - [Access SpatialMQA](#2-Access-SpatialMQA)
    - [Download Images](#Download-Images)
    - [Data Split](#Data-Split)
    - [Data Format](#Data-Format)
  - [Experiment & Evaluation](#3-Experiment-and-Evaluation)
    - [Experiment](#Experiment)
    - [Evaluation](#Evaluation)
  - [License](#4-License)




## 1 Overview
**SpatialMQA** is a **manually annotated** dataset designed for **multimodal spatial relation reasoning** in a **m**ultiple-choice **q**uestion & **a**nswer format.
The dataset includes 5,392 samples collected from COCO2017, covering 128 subject and object types, without bounding boxes. To address the limitations of existing datasets, we clearly define annotation guidelines for SpatialMQA, including standardizing the objective world as the coordinate system and avoiding questions that can be answered solely by the question itself. 

### Examples
The following figures list some classic examples in our dataset. You can click out [`Examples:1~4/`](Examples/examples_1-4.png) and [`Examples:5~8/`](Examples/examples_5-8.png) to view the details.

### Detail Information
The following table [`Splits/`](Comparison/splits.png) lists the detailed information statistics of the splited dataset.
<br>
You can find our dataset through the following path **_(Dataset/dataset)_** for more details.
<br>
_Due to the fact that only redirecting to the specified file is valid in anonymous links, redirecting to the specified directory is invalid. Therefore, we use bold and italicized font to indicate the markings of all specified directories, making it easier for reviewers to search. Thank you!_


## 2 Access SpatialMQA
### Download Images
We use a subset of COCO-2017's images. The following script download COCO-2017's test sets images then put them into a single fodler `Dataset/COCO2017/`.

```bash
cd Dataset/ 
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mv test2017 COCO2017 && rm -r test2017
```
Copy only relevant images to `relevant_images/`.
```bash
mkdir relevant_images
cd tool
python select_revlevant_images.py
```
Alternatively, you could also browse individual images online directly using the key "image" in single json data.
<br>(Through COCO's open source link, 'http://images.cocodataset.org/test2017/' + 'image_name'. For example: http://images.cocodataset.org/test2017/000000195921.jpg.)

###  Data Split
As reported in the folloeing table, SpatialMQA contains 5,392 samples, divided into training, validation, and test sets according to a 7:1:2 ratio.
<br>All the splited data sets are in the directory **_(Dataset/dataset)_**. 

### Data Format
Each `jsonl` file is of the following format:
```json
{"image": "000000000933.jpg", "question": "Where is the fork located relative to the pizza?", "options": ["on/above", "below", "in front of", "behind", "left of", "right of"], "answer": "right of"}
{"image": "000000100633.jpg", "question": "If you are the cyclist in the image, where is the dog located relative to you?", "options": ["in front of", "behind", "left of", "right of"], "answer": "behind"}
{"image": "000000070986.jpg", "question": "If you are the driver of the bus in the image, from your perspective, where is the red car located relative to the bus?", "options": ["in front of", "behind", "left of", "right of"], "answer": "left of"}
{"..."}
```
Each line is an individual data point.
`image` denotes name of the image in COCO. `question` is the question with manual annotation, `options` is reasonable combinations of six spatial relationships:(on/above, below, in front of, behind, left of, right of. `answer` is the annotation based on the objective world.
<br>
Our dataset is expanded based on the categories included in the COCO dataset. There are 113 subject types and one additional type for subjects with five or fewer samples in our dataset, and 84 object types and one additional type for objects with five or fewer samples. Due to the overlap between subject and object types, we have a total of 128 distinct subject and object types. You can see all of them in the file [`S & O types/`](Dataset/types/types.txt). 


## 3 Experiment and Evaluation
### Experiment
We have disclosed the inference code for the model in the directory **_(Code/experiment)_**,  as well as the fine-tuning code in the directory **_(Code/finetune)_**.
<br>
- For all 6 open-sourse MLLMs, you can directly execute Python files in the directory **_(Code/experiment)_** to perform inference on models before and after fine-tuning: 
```
nohup python blip-vqa-base.py > log/blip_exp.log 2>1& &
nohup python blip-vqa-base_finetuned.py > log/blip_finetuned_exp.log 2>1& &
nohup python blip2-opt-2.7b.py > log/blip2_exp.log 2>1& &
nohup python blip2-lora.py > log/blip2_lora_exp.log 2>1& &
nohup python instructblip-flan-t5-xl.py > log/instructblip_exp.log 2>1& &
nohup python instructblip-lora.py > log/instructblip_lora_exp.log 2>1& &
nohup python idefics_new.py > log/idefics_exp.log 2>1& &
nohup python idefics_lora.py > log/idefics_lora_exp.log 2>1& &
nohup python spatial_test_llava.py > log/llava_exp.log 2>1& &
nohup python spatial_test_llava_lora.py > log/llava_lora_exp.log 2>1& &
nohup python spatial_test_mplug.py > log/mplug_exp.log 2>1& &
nohup python spatial_test_mplug_lora.py > log/mplug_lora_exp.log 2>1& &
```
Due to the large amount of open source model code, you need to download it yourself through channels or call it directly from platforms such as [huggingface](https://huggingface.co).
<br>
- For blip, blip2, instructblip and idefics, you can directly execute Python files in the directory **_(Code/finetune)_** to perform fine-tuning: 
```
nohup python blip-vqa-base.py > log/blip_train.log 2>1& &
nohup python blip2-lora.py > log/blip2_train.log 2>1& &
nohup python instructblip-lora.py > log/instructblip_train.log 2>1& &
nohup python idefics.py > log/idefics_train.log 2>1& &
```
- For llava and mplug-owl, you need to execute bash files in the directory **_(Code/finetune)_** to perform fine-tuning:
```
nohup bash llava_lora_train.sh > log/llava_train.log 2>1& &
nohup bash mPLUG_Owl_train_it.sh > log/mplug_train.log 2>1& &
```
- For gemini-pro-v and gpt-4v, you can directly execute our Python file in the directory **_(Code/close_models)_** to perform inferencing of the zero-shot, few-shot and text-only, provided that you prepare a key:
```
python gemini_text_only.py
python gemini_zero_shot.py
python gemini_1_shot.py
python gemini_2_shot.py
python gemini_3_shot.py
python gpt4_text_only.py
python gpt4_zero_shot.py
python gpt4_1_shot.py
python gpt4_2_shot.py
python gpt4_3_shot.py
```
Gemini needs to apply on the [official website](https://aistudio.google.com/app/apikey), and GPT4 needs to be purchased on the [official website](https://openai.com/).

### Evaluation
You can process the results of model inference through the code we provide to calculate overall accuracy, overall P, R, F1 indicators, accuracy based on relationship categories, and accuracy based on rules. We integrate the calculation process into the Python files in the directory **_(Code/eval)_**:
```
python calculate_prf1.py
python calculate_xyz.py
python calculate_result_rule.py
```

### Requirements
The environment configuration required for debugging code is placed in directory **_(Code/requirement)_**
<br>
The requirements of models blip, blip2 and instructblip, are all in the file [`requirement_blip.txt/`](Code/requirement/requirement_blip.txt)

## 4 License
This project is licensed under the [Apache-2.0 License](LICENSE).
