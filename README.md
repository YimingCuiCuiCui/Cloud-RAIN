# Cloud-RAIN: Point Cloud Analysis With Reflectional-Invariance

This repository is an official implementation of the paper "Cloud-RAIN: Point Cloud Analysis With Reflectional-Invariance". Code will be released soon, stay tuned!

**Tip:** The result of point cloud experiment usually faces greater randomness than 2D image. We suggest you run your experiment more than one time and select the best result.

&nbsp;
## Point Cloud Semantic Segmentation on the S3DIS benchmark
You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under `data/`

### Run the training script:

This task use 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. 

- Train in area 1,2,3,4,6
```
python main_semseg_s3dis.py --exp_name=EXP_NAME --test_area=5 --model MODEL_NAME
```
Example:
```
python main_semseg_s3dis.py --exp_name=dgcnn_semseg_5_aug_no_norm --test_area=5 --model dgcnn
```


### Run the evaluation script with pretrained models:

- Evaluate in area 5

``` 
python main_semseg_s3dis.py --exp_name=EXP_NAME--test_area=5 --model MODEL_NAME--eval True --model_root MODEL_ROOT
```
Example:
```
python main_semseg_s3dis.py --exp_name=dgcnn_semseg_5 --test_area=5 --model dgcnn --eval True --model_root ./checkpoints/dgcnn_semseg_5/models/
```

&nbsp;
## Point Cloud Semantic Segmentation on the ScanNet benchmark

### Run the training script:
```
python main_semseg_scannet.py --exp_name=EXP_NAME --model MODEL_NAME
```
Example:
```
python main_semseg_scannet.py --exp_name=pointnet_semseg_scannet  --model pointnet
```

