import json
import os
import random


def convert_json_to_target_format(cot_result_path, rm_result_path, output_file, data_key):
    # 加载数据
    with open(cot_result_path, 'r') as f1:
        cot_data = json.load(f1)

    rm_data = []
    with open(rm_result_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rm_data.append(data)

    json_list = []

    for item1, item2 in zip(cot_data[data_key], rm_data):
        if not (item2.get('chosen') == item1["response1"] or item2.get('chosen') == item1["response2"]):
            raise ValueError("The data is not matched!")
        if (item2.get('results') == 1 and item1["result"]["orig"]["is_correct"] == True) or (item2.get('results') == 0 and item1["result"]["orig"]["is_correct"] == False):
            if item1["result"]["orig"]["prediction"] == 1:
                chosen, rejected = "Output (a)", "Output (b)"
            elif item1["result"]["orig"]["prediction"] == 2:
                chosen, rejected = "Output (b)", "Output (a)"
            else:
                continue
            output = item1["result"]["orig"]["completion"]
        elif (item2.get('results') == 1 and item1["result"]["swap"]["is_correct"] == True) or (item2.get('results') == 0 and item1["result"]["swap"]["is_correct"] == False):
            item1["response1"], item1["response2"] = item1["response2"], item1["response1"]
            if item1["result"]["swap"]["prediction"] == 1:
                chosen, rejected = "Output (a)", "Output (b)"
            elif item1["result"]["swap"]["prediction"] == 2:
                chosen, rejected = "Output (b)", "Output (a)"
            else:
                continue
            output = item1["result"]["swap"]["completion"]
        else:
            continue


        # Randomly swap the outputs
        if random.choice([True, False]):
            chosen, rejected = rejected, chosen
            output = output.replace(chosen, "【IMCRAZY】")
            output = output.replace(rejected, chosen)
            output = output.replace("【IMCRAZY】", rejected)
            item1["response1"], item1["response2"] = item1["response2"], item1["response1"]
        json_object = {
            "instruction": f"""As an evaluation expert, given an instruction and its two possible outputs, compare the outputs. Below are the instruction and its candidate outputs:

# Instruction:
{item1["instruction"]}

# Output (a):
{item1["response1"]}

# Output (b):
{item1["response2"]}


Given that {chosen} is better than {rejected}, please provide the rationale and end with "Therefore, {chosen} is better." (Do NOT output any other words.):""",
            "input": "",
            "system": """You are an evaluation expert. Your goal is to provide the rationale based on the given evaluation result.""",
            "output": output
        }
        # Append the converted data to the list
        json_list.append(json_object)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Write the converted data to the output file in JSON format
    with open(output_file, 'w') as file:
        json.dump(json_list, file, indent=4)


cot_result_path = '../result/Llama-3.2-3B-Instruct/arena.json'
rm_result_path = '../result/arena/output/Llama-3.2-3B-Instruct_test_len1024_fulltrain_1e-05_dataarena_dpo.json_outputs.jsonl'
# The output file is used to train rationale generation model
cot_output_file_path = '../data/arena_critique_distill.json'

data_key = 'arena'

convert_json_to_target_format(
    cot_result_path, rm_result_path, cot_output_file_path, data_key)

print("Over!")

