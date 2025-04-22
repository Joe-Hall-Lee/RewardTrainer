import json


def convert_json_data(json_data):
    result = []
    for item in json_data:
        if item["turn"] != 1:
            continue
        instruction = item["conversation_a"][0]["content"]
        answer_0 = item["conversation_a"][1]["content"]
        answer_1 = item["conversation_b"][1]["content"]

        if item["winner"] == "model_a":
            chosen = answer_0
            rejected = answer_1
        elif item["winner"] == "model_b":
            chosen = answer_1
            rejected = answer_0
        else:
            continue

        result.append({
            "instruction": instruction,
            "chosen": chosen,
            "rejected": rejected
        })
    return result


def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


json_data = load_json_data(
    "F:\\CS\\AI\\JudgePO\\data\\arena.json")
result = convert_json_data(json_data)

with open("F:\\CS\\AI\\JudgePO\\LLaMA-Factory\\data\\arena_dpo.json", "w") as f:
    json.dump(result, f, indent=4)
