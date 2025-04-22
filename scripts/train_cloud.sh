export CUDA_VISIBLE_DEVICES=1,2,4,5

composer -n 4 train/cloud/train/train.py train/cloud/train/configs/8b_cloud.yaml