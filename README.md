<br />
<p align="center">
  <h1 align="center"> 🔭 Can Multimodal Large Language Models Understand Spatial Relations</h1>
  <h3 align="center">SpatialMQA: A new dataset for spatial reasoning of MLLMs.</h3>
  
  <p align="center">  
<!--     <a href="https://arxiv.org/abs/2205.00363">arxiv</a> -->
    ·
    <a href="https://github.com/ziyan-xiaoyu/SpatialMQA/blob/master/Dataset">dataset</a>
    ·
<!--     <a href="https://paperswithcode.com/sota/visual-reasoning-on-vsr">benchmark</a> -->
    
  </p>
</p>


## Contents

- [ImgFact](#Contents)
  - [Overview](#1-Overview)
    - [Examples](#Examples)
    - [Detail information](#Detail-Information)
  - [Access SpatialMQA](#2-Access-SpatialMQA)
    - [Download images](#Download-Images)
    - [Splits of the data](#Data-Split)
    - [Format of the data](#Data-Format)
  - [Experiment & Evaluation](#4-Experiment-&-Evaluation)
    - [Experiment](#Experiment)
    - [Evaluation](#Evaluation)
  - [License](#4-License)




## 1 Overview
**SpatialMQA is a manually annotated dataset designed for multimodal spatial relation reasoning in a multiple-choice question & answer format.**
The dataset includes 5,392 samples collected from COCO2017, covering 128 subject and object types, without bounding boxes. To address the limitations of existing datasets, we clearly define annotation guidelines for SpatialMQA, including standardizing the objective world as the coordinate system and avoiding questions that can be answered solely by the question itself. 

### Examples
The following figures list some classic examples in our dataset. You can click out [`Examples:1~4/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Examples/examples_1-4.png) and [`Examples:5~8/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Examples/examples_5-8.png) to view the details.

### Detail Information
The following table [`Splits/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Comparison/splits.png) lists the detailed information statistics of the splited dataset.
<br>
Check out [`dataset/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Dataset) for more details.


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
<br>All the splited data sets are in the directory [`dataset/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Dataset). 

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
Our dataset is expanded based on the categories included in the COCO dataset. There are 113 subject types and one additional type for subjects with five or fewer samples in our dataset, and 84 object types and one additional type for objects with five or fewer samples. Due to the overlap between subject and object types, we have a total of 128 distinct subject and object types. You can see all of them in the file [`S & O types/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Dataset/types/types.txt). 


## 3 Experiment & Evaluation
### Experiment
We have disclosed the inference and fine-tuning code for the model [`experiment/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Code/experiment), as well as the code required for evaluation [`eval/`](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/Code/eval).
<br>
- For blip, blip2, instructblip and ideficts, you can directly execute Python files: 
```
nohup python filter_tuples.py > log/.log 2>1& &
python gen_sample_tuples.py
python gen_candidate_relations.py
python gen_visual_relations.py
```
- For llava and mplug-owl, you need to execute bash files:
```
python filter_tuples.py
python gen_sample_tuples.py
python gen_candidate_relations.py
python gen_visual_relations.py
```
Due to the large amount of open source model code, you need to download it yourself through channels or call it directly from platforms such as [huggingface](https://huggingface.co).
<br>
- For gemini-pro-v and gpt-4v, you can directly execute our Python file, provided that you prepare a key:
```
python filter_tuples.py
python gen_sample_tuples.py
python gen_candidate_relations.py
python gen_visual_relations.py
```
Gemini needs to apply on the [official website](https://aistudio.google.com/app/apikey), and GPT4 needs to be purchased on the [official website](https://openai.com/).

### Evaluation
You can process the results of model inference through the code we provide to calculate overall accuracy, overall P, R, F1 indicators, accuracy based on relationship categories, and accuracy based on rules. We integrate the calculation process into the following Python files：
```
python filter_tuples.py
python gen_sample_tuples.py
python gen_candidate_relations.py
python gen_visual_relations.py
```



## 4 License
This project is licensed under the [Apache-2.0 License](https://github.com/ziyan-xiaoyu/SpatialMQA/blob/master/LICENSE).
