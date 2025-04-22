import json
import re

def remove_templates(text):
    """移除模板标记，例如<|...|>"""
    return re.sub(r'<\|.*?\|>', '', text).strip()

def process_file(input_filename, output_filename):
    output_data = []
    with open(input_filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # 提取 instruction
            prompt = data.get('prompt', '')
            if "llama-3" in input_filename.lower():
                instruction_match = re.search(
                    r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
                    prompt,
                    re.DOTALL
                )
            elif "mistral" in input_filename.lower():
                instruction_match = re.search(
                    r'<s>\s*\[INST\]\s*(.*?)\s*\[/INST\]',
                    prompt,
                    re.DOTALL
                )
            if instruction_match:
                instruction = remove_templates(instruction_match.group(1))
            else:
                instruction = ''
            
            # 处理 chosen 和 rejected
            chosen_text = remove_templates(data.get('chosen', ''))
            rejected_text = remove_templates(data.get('rejected', ''))
            
            # 根据 results 字段确定是否需要交换 chosen 和 rejected
            if data.get('results', 1) == 0:
                chosen_text, rejected_text = rejected_text, chosen_text
            
            # 创建新的数据条目
            output_entry = {
                'instruction': instruction,
                'chosen': chosen_text,
                'rejected': rejected_text
            }
            output_data.append(output_entry)
    
    # 将结果写入输出文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_file = 'result/helpsteer2/output/Llama-3-8B-Instruct_test_len1024_fulltrain_1e-05_datahelpsteer2-dpo.json_outputs.jsonl'
    output_file = 'data/helpsteer2-dpo_llama.jsonl'

    process_file(input_file, output_file)
    print(f"转换完成，结果已保存到 {output_file}")
