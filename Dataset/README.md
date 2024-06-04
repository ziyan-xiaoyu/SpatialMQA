# Data of SpatialMQA

### Download images
We use a subset of COCO-2017's images. The following script download COCO-2017's test sets images then put them into a single fodler `COCO2017/`.

```bash
cd data/ # enter this folder 
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

Alternatively, you could also browse individual images online directly through COCO's open source link using the key "image" in our dataset (For example: http://images.cocodataset.org/test2017/000000195921.jpg.)

### Splits
As reported in the folloeing table, SpatialMQA contains 5,392 samples, divided into training, validation, and test sets according to a 7:1:2 ratio.<br>
All the splited data sets are in the directory [[dataset]](Dataset/dataset).
![](Comparision/splits.jpg) 

### Format of the data
Each `jsonl` file is of the following format:
```json
{"image": "000000000933.jpg", "question": "Where is the fork located relative to the pizza?", "options": ["on/above", "below", "in front of", "behind", "left of", "right of"], "answer": "right of"}
{"image": "000000100633.jpg", "question": "If you are the cyclist in the picture, where is the dog located relative to you?", "options": ["in front of", "behind", "left of", "right of"], "answer": "behind"}
{"image": "000000070986.jpg", "question": "If you are the driver of the bus in the picture, from your perspective, where is the red car located relative to the bus?", "options": ["in front of", "behind", "left of", "right of"], "answer": "left of"}
{"..."}
```
Each line is an individual data point.
`image` denotes name of the image in COCO. `question` is the question with manual annotation, `options` is reasonable combinations of six spatial relationships:(on/above, below, in front of, behind, left of, right of. `answer` is the annotation based on the objective world.

