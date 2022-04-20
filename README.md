## DirectionNet

This repository will contain the TensorFlow code for the model introduced in the CVPR 2020 paper:

**Wide-Baseline Relative Camera Pose Estimation with Directional Learning** \
Kefan Chen, Noah Snavely, Ameesh Makadia \
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020)*. \
[arXiv](https://arxiv.org/abs/2106.03336)


Required packages: tensorflow 1.15, tensorflow_graphics, tensorflow_addons, tensorflow_probability, tf_slim, pickle

MatterportA test data: https://drive.google.com/file/d/1be75Ys8vi1o7eeS_Rf0SuJxlTkDJNisZ/view?usp=sharing
MatterportB test data: https://drive.google.com/file/d/1PcyD_8TZOOKh6G8B8eUHQrOUEOMrMx_F/view?usp=sharing
Checkpoints trained on MatterportA: https://drive.google.com/file/d/1ATA1-FwWb_sKAV4uWcpj7ZrMu59ZhG3_/view?usp=sharing

dataset.py:
	generate_from_meta can create the datasets of images and ground truth from the Matterport3D dataset given the meta data files.

	generate_random_views can be used to generate a large-scale wide stereo dataset with camera pose labels from a panoramic image dataset.