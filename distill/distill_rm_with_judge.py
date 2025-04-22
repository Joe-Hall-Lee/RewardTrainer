import json
import os
import random

def convert_json_to_target_format(input_file, output_file, data_key):
    # Prepare an empty list to store the converted data
    json_list = []

    # Open the JSON file and read its contents
    with open(input_file, 'r') as file:
        data = json.load(file)
        for item in data["data_key"]:
            if item["result"]["orig"]["prediction"] == 1:
                chosen = item["response1"]
                rejected = item["response2"]
            elif item["result"]["orig"]["prediction"] == 2:
                chosen = item["response2"]
                rejected = item["response1"]
            else:
                if item["label"] == 1:
                    chosen = item["response1"]
                    rejected = item["response2"]
                elif item["label"] == 2:
                    chosen = item["response2"]
                    rejected = item["response1"]
            json_object = {
                "instruction": item["instruction"],
                "chosen": chosen,
                "rejected": rejected
            }
            # Append the converted data to the list
            json_list.append(json_object)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    # Write the converted data to the output file in JSON format
    with open(output_file, 'w') as file:
        json.dump(json_list, file, indent=4)


input_file_path = '../result/Llama-3.2-3B-Instruct-hh-rlhf_et_cot_distill_lr_1e-5_epoch_3/hh-rlhf.json'
output_file_path = '../data/hh-rlhf_rm_star_distill.json'
data_key = "hh-rlhf"

convert_json_to_target_format(input_file_path, output_file_path, data_key)

print("Over!")
