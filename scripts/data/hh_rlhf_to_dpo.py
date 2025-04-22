import json


def process_file(input_file_path, output_file_path, sample_size=20000):
    # 存储所有数据
    all_data = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                # 获取 chosen 和 rejected 的内容
                chosen = data.get('chosen', '').strip()
                rejected = data.get('rejected', '').strip()

                # 过滤掉 chosen 和 rejected 相同的条目
                if chosen == rejected:
                    continue

                # 过滤掉多轮对话（多个 "Human:" 或 "Assistant:"）
                if has_multiple_turns(chosen) or has_multiple_turns(rejected):
                    continue

                # 提取 Human 和 Assistant 的回答
                instruction = extract_instruction(chosen)
                chosen_response = extract_assistant_response(chosen)
                rejected_response = extract_assistant_response(rejected)

                # 计算总字符数
                total_length = len(chosen) + len(rejected)

                # 将数据和长度一起存入列表
                all_data.append({
                    "instruction": instruction,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "total_length": total_length
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue


    all_data.sort(key=lambda x: x["total_length"])
    shortest_data = all_data[20000: 20000 + sample_size]

    # 只提取数据，不包含 total_length
    result_data = [{"instruction": item["instruction"], "chosen": item["chosen"],
                    "rejected": item["rejected"]} for item in shortest_data]

    # 写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(result_data, outfile, ensure_ascii=False, indent=4)

    print(
        f"Processing complete. {len(result_data)} entries written to {output_file_path}.")


def extract_instruction(conversation):
    """
    Extract the instruction from the conversation history.
    Assumes that the instruction is the first line starting with 'Human:'
    """
    for line in conversation.split('\n'):
        if line.strip().startswith('Human:'):
            return line.replace('Human:', '').strip()
    return "No instruction found"


def extract_assistant_response(conversation):
    """
    Extract the Assistant's response from the conversation history.
    Assumes that the response follows a line starting with 'Assistant:'
    """
    for line in conversation.split('\n'):
        if line.strip().startswith('Assistant:'):
            return line.replace('Assistant:', '').strip()
    return "No Assistant response found"


def has_multiple_turns(conversation):
    """
    Check if the conversation contains multiple turns (i.e., more than one "Human:" and "Assistant:").
    Returns True if there are multiple turns, False if it's a single round of conversation.
    """
    human_count = sum(1 for line in conversation.split('\n')
                      if line.strip().startswith('Human:'))
    assistant_count = sum(1 for line in conversation.split(
        '\n') if line.strip().startswith('Assistant:'))

    # If there are more than one Human and/or Assistant, we consider it as multiple turns
    return human_count > 1 or assistant_count > 1


if __name__ == "__main__":
    input_file = '../../data/hh-rlhf.json'  # 输入文件路径
    output_file = '../../data/hh-rlhf_dpo.json'  # 输出文件路径
    sample_size = 20000  # 取 20000 条数据

    process_file(input_file, output_file, sample_size)

