export CUDA_VISIBLE_DEVICES=7
python train/cloud/eval/eval.py \
    --model-path output/Llama-3-8B-Instruct-Cloud-HHRLHF \
    --benchmark data/rewardbench/filtered.json \
    --batch-size 8