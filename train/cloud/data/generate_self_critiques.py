import os
from argparse import ArgumentParser

from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from train.cloud.train.train import COT_PROMPT
from train.cloud.train.data import build_chat_messages

def build_feedback_prompts(tokenizer, example):
    bos_text = tokenizer.decode([tokenizer.bos_token_id])
    eos_text = tokenizer.decode([tokenizer.eos_token_id])
    eot_text = "<|eot_id|>"  # Hard coded for llama3 end of turn id for now but oh well

    chosen_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["chosen"]), tokenize=False)
    rejected_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["rejected"]), tokenize=False)
    cot_fmt = tokenizer.apply_chat_template([{"role": "user", "content": COT_PROMPT}], tokenize=False).replace(bos_text, "").replace(eos_text, "").replace(eot_text, "")

    example["chosen_feedback_prompt"] = chosen_prefix + cot_fmt
    example["rejected_feedback_prompt"] = rejected_prefix + cot_fmt
    return example

def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize vLLM model
    llm = LLM(model=args.model, max_model_len=args.max_tokens)

    # Configure sampling parameters
    eot_text = "<|eot_id|>"
    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        stop=[eot_text, tokenizer.eos_token]  # Stop tokens
    )

    # Load the dataset from local file
    file_extension = os.path.splitext(args.base_dataset)[1].lower()
    if file_extension == ".json" or file_extension == ".jsonl":
        dataset_format = "json"
    elif file_extension == ".csv":
        dataset_format = "csv"
    elif file_extension == ".parquet":
        dataset_format = "parquet"
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .json, .jsonl, .csv, .parquet")

    # Load dataset from local file
    ds = load_dataset(dataset_format, data_files=args.base_dataset, split="train")
    
    ds = ds.map(lambda x: build_feedback_prompts(tokenizer, x), num_proc=10)

    def fetch_response(examples):
        chosen_feedback_prompts = [example["chosen_feedback_prompt"] for example in examples]
        rejected_feedback_prompts = [example["rejected_feedback_prompt"] for example in examples]

        # Generate chosen feedback
        chosen_outputs = llm.generate(chosen_feedback_prompts, sampling_params)
        chosen_feedback = [output.outputs[0].text for output in chosen_outputs]

        # Generate rejected feedback
        rejected_outputs = llm.generate(rejected_feedback_prompts, sampling_params)
        rejected_feedback = [output.outputs[0].text for output in rejected_outputs]

        results = [
            {**example, "chosen_feedback": [chosen], "rejected_feedback": [rejected]}
            for example, chosen, rejected in zip(examples, chosen_feedback, rejected_feedback)
        ]
        return results

    all_feedback = []
    bs = 128
    for i in tqdm(range(0, len(ds), bs), desc="Fetching responses", leave=True):
        batch_examples = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        all_feedback.extend(fetch_response(batch_examples))

    hf_feedback_ds = Dataset.from_list(all_feedback)
    cols_to_select = ["prompt", "chosen", "rejected", "chosen_feedback", "rejected_feedback", "id"]
    hf_feedback_ds = hf_feedback_ds.select_columns(cols_to_select)

    # Save to local JSON file as an array
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    hf_feedback_ds.to_json(args.output, orient="records", lines=False, force_ascii=False, indent=2)
    print(f"Dataset saved to {args.output}")

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Model / data params
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model identifier (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--base-dataset", type=str, required=True, help="Path to local dataset file (e.g., .json, .jsonl, .csv, .parquet)")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON file")

    # Sampling params
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args()

    main(args)