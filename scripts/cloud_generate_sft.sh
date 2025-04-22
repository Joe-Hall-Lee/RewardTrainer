export CUDA_VISIBLE_DEVICES="4" 

python /H1/zhouhongli/STaR-Judge/train/cloud/data/generate_self_critiques.py --model output/Llama-3-8B-Instruct-Critique-SFT-HHRLHF --base-dataset /H1/zhouhongli/STaR-Judge/data/hhrlhf-cloud-dpo.json --output /H1/zhouhongli/STaR-Judge/data/hhrlhf_critique_llama3.json