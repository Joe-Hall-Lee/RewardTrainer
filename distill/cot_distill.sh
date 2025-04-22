#!/bin/bash
#SBATCH --job-name=Llama-2-13b-chat_lr_2e-5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --gres=gpu:1  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus # 分区默认即可

INPUT="../result/arena/output/Llama-3.2-3B-Instruct_test_len1024_fulltrain_1e-05_dataarena_rm_llama.jsonl_outputs.jsonl"
OUTPUT_FILE="../data/arena_cot_distill_verified.json"
MODEL_NAME="../output/Llama-3.2-3B-Instruct-arena_critique_distill_lr_1e-5_epoch_3"
TEMPERATURE=0.0
MAX_TOKENS=512

CUDA_VISIBLE_DEVICES="5" python cot_distill.py \
    --input "$INPUT" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS"
