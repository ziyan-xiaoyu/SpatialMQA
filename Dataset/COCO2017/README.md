### Download images
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
<br>(Through COCO's open source link, http://images.cocodataset.org/test2017/ + 'image_name'. For example: http://images.cocodataset.org/test2017/000000195921.jpg.)
