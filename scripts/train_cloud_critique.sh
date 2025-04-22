export CUDA_VISIBLE_DEVICES=2,3,4,5

composer -n 4 train/cloud/train/train.py train/cloud/train/configs/8b_critique_sft.yaml