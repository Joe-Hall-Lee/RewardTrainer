import json


def convert_json_array_to_json(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    json_list = []

    for item in data:
        output = item["chosen"]

        json_object = {
            "instruction": item["instruction"],
            "input": "",
            "system": "",
            "output": output
        }
        json_list.append(json_object)

    with open(output_file, 'w') as file:
        json.dump(json_list, file, indent=4)


input_file_path = 'F:/CS/AI/JudgePO/LLaMA-Factory/data/arena_dpo.json'
output_file_path = 'F:/CS/AI/JudgePO/LLaMA-Factory/data/arena_sft.json'

convert_json_array_to_json(input_file_path, output_file_path)
print("Conversion complete.")
