## DirectionNet

This repository will contain the TensorFlow code for the model introduced in the CVPR 2020 paper:

**Wide-Baseline Relative Camera Pose Estimation with Directional Learning** \
Kefan Chen, Noah Snavely, Ameesh Makadia \
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020)*. \
[arXiv](https://arxiv.org/abs/2106.03336)

## Requirements

Required packages: tensorflow 1.15, tensorflow_graphics, tensorflow_addons, tensorflow_probability, tf_slim, pickle

## Dataset and Pre-trained Models

[MatterportA test data](https://drive.google.com/file/d/1be75Ys8vi1o7eeS_Rf0SuJxlTkDJNisZ/view?usp=sharing)\
[MatterportB test data](https://drive.google.com/file/d/1PcyD_8TZOOKh6G8B8eUHQrOUEOMrMx_F/view?usp=sharing)
<!-- [MatterportA Checkpoints](https://drive.google.com/file/d/1ATA1-FwWb_sKAV4uWcpj7ZrMu59ZhG3_/view?usp=sharing)\
[MatterportB Checkpoints](https://drive.google.com/file/d/14OUSXnay8VD5rARxXwwLX11z-ScibXN8/view?usp=sharing) -->

1. dataset.generate_from_meta can create the datasets of images and ground truth from the Matterport3D dataset given the meta data files.

2. dataset.generate_random_views can be used to generate a large-scale wide stereo dataset with camera pose labels from a panoramic image dataset.

## Train DirectionNet-R and DirectionNet-T

1. train DirectionNet-R.
```
python train.py \
--checkpoint_dir <path_to_checkpoints_and_logs> \
--data_dir <path_to_training_set> \
--model 9D
```
2. Run DirectionNet-R on the training and test sets, then save the estimated rotations as Python pickle dictionary in the data directories respectively.
3. train DirectionNet-T.
```
python train.py \
--checkpoint_dir <path_to_checkpoints_and_logs> \
--data_dir <path_to_training_set> \
--model T
```

## Evaluation
DirectionNet-R
```
python eval.py \
--checkpoint_dir <path_to_load_checkpoints> \
--eval_data_dir <path_to_test_set> \
--save_summary_dir <path_to_save_logs> \
--testset_size <testset_size> \
--batch <test_batch> \
--model 9D
```

DirectionNet-T
```
python eval.py \
--checkpoint_dir <path_to_load_checkpoints> \
--eval_data_dir <path_to_test_set> \
--save_summary_dir <path_to_save_logs> \
--testset_size <testset_size> \
--batch <test_batch> \
--model T
```

## Citation
```
@InProceedings{Chen_2021_CVPR,
author    = {Chen, Kefan and Snavely, Noah and Makadia, Ameesh},
title     = {Wide-Baseline Relative Camera Pose Estimation With Directional Learning},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2021},
pages     = {3258-3268}
}
```
