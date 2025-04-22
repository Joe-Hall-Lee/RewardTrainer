import datasets

# 加载数据集
ds = datasets.load_dataset("Anthropic/hh-rlhf", cache_dir="../data")

# 将数据集保存为 JSON 文件
ds["train"].to_json("../data\\hh-rlhf.json", orient="records", lines=True)

