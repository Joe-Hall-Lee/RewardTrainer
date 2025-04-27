export CUDA_VISIBLE_DEVICES=3
python train/cloud/eval/eval.py \
    --model-path output/Mistral-7B-Instruct-v0.3-CLoud-HelpSteer2 \
    --benchmark data/rewardbench/filtered.json \
    --batch-size 8