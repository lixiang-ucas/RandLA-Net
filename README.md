This is the source code for our paper <b></b>.

# A Novel Semi-Supervised Method for Airborne LiDAR Point Cloud Classification


Introduction
------------
This is the source code for our paper **A Novel Semi-Supervised Method for Airborne LiDAR Point Cloud Classification**. This repo includes the code for conducting experiments using RandLA-Net as the backbone network.

## Data Preparation:

python utils/data_prepare_isprs_large_w_height_mask.py

python utils/data_prepare_dfc.py

## ISPRS dataset

training:
python main_isprs.py --mode train --gpu 0

evaluation:
python main_isprs.py --mode test --gpu 0


## DFC 3D dataset

training:
python main_dfc.py --mode train --gpu 0

evaluation:
python main_dfc.py --mode test --gpu 0
