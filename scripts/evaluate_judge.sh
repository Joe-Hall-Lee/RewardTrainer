#!/bin/bash
#SBATCH --job-name=Mistral-7B-Instruct-v0.3-helpsteer2_et_lr_1e-6_eval  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可

export CUDA_VISIBLE_DEVICES=3
python eval/run_bench.py \
    --name Llama-3.2-3B-Instruct-arena_cot_distill_verified_lr_1e-5_epoch_3 \
    --config configs/eval/llama3-3b.yaml \
    --benchmark rewardbench \
    --data-path data
