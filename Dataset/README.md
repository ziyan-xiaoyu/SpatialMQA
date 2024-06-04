# Data 

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

Alternatively, you could also browse individual images online through COCO's open source link using the key "image" in our dataset (For example: http://images.cocodataset.org/test2017/000000195921.jpg.)

### Splits
As reported in the folloeing table, SpatialMQA contains 5,392 samples, divided into training, validation, and test sets according to a 7:1:2 ratio.<br>
All the splited data sets are in the directory [[dataset]](Dataset/dataset).
![](Comparision/splits.jpg) 

### Format of the data
Each `jsonl` file is of the following format:
```json
{"image": "000000050403.jpg", "image_link": "http://images.cocodataset.org/train2017/000000050403.jpg", "caption": "The teddy bear is in front of the person.", "label": 1, "relation": "in front of", "annotator_id": 31, "vote_true_validator_id": [2, 6], "vote_false_validator_id": []}
{"image": "000000401552.jpg", "image_link": "http://images.cocodataset.org/train2017/000000401552.jpg", "caption": "The umbrella is far away from the motorcycle.", "label": 0, "relation": "far away from", "annotator_id": 2, "vote_true_validator_id": [], "vote_false_validator_id": [2, 9, 1]}
{"..."}
```
Each line is an individual data point.
`image` denotes name of the image in COCO and `image_link` points to the image on the COCO server (so you can also access directly). `caption` is self-explanatory. `label` being `0` and `1` corresponds to False and True respectively. `relation` records the spatial relation used. `annotator_id` points to the annotator who originally wrote the caption. `vote_true_validator_id` and `vote_false_validator_id` are annotators who voted True or False in the second phase validation.

### Other data files
[`data_files/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files) contain the major data collected for creating VSR. [`data_files/all_vsr_raw_data.jsonl`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/all_vsr_raw_data.jsonl) contains all 12,809 raw data points and [`data_files/all_vsr_validated_data.jsonl`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/all_vsr_validated_data.jsonl) contains the 10,119 data points that passed the second-round validation (and is used for creating the random and zeroshot splits). [`data_files/meta_data.csv`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/meta_data.jsonl) contains meta data of annotators.
