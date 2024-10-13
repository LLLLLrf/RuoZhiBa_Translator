import json
import numpy as np
import os

# 加载和预处理数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    original_sentences = [entry['original_data'] for entry in data]
    annotated_results = [entry['annotated_result_1'] for entry in data]  # 选一个annotated_result?
    return original_sentences, annotated_results

def load_files_names(mode="train"):
    # 读取data目录下以train和val开头的json文件
    if mode == "train":
        file_names = [file_name for file_name in os.listdir('data') if file_name.startswith('train')]
    elif mode == "val":
        file_names = [file_name for file_name in os.listdir('data') if file_name.startswith('val')]
    else:
        raise ValueError('mode should be "train" or "val"')

    return file_names
    