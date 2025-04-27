export CUDA_VISIBLE_DEVICES="5" 

python train/cloud/data/generate_self_critiques.py --model output/Llama-3-8B-Instruct-Critic-HHRLHF --base-dataset data/hhrlhf-skywork-cloud-dpo.json --output data/hhrlhf-skywork_critique_llama.json