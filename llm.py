import requests
import json
import time
from torch.utils.data import DataLoader, Dataset

# Qwen API 及 GPT-4o-mini API的URL和密钥
qwen_api_url = "https://api.qwen.ai/generate"
gpt_4o_mini_url = "https://api.gpt-4o-mini.ai/generate"
api_key =''
api_id=''
model_name = 'qwen'  # 或 'gpt-4o-mini'

# In-Context Learning 示例数据
# 数据样例
in_context_data = [
    {
        "original_data": "谁能想到，这张仿佛温和无害的笑脸之下竟然隐藏着面部肌肉和神经",
        "annotated_results": [
            "这句话强调的是，人类的表面情感（如微笑）背后隐藏着复杂的生理和情感机制。",
            "笑脸通常会伴随面部肌肉和神经活动，但是这个描述很怪",
            "这句话用幽默的方式将看似温和无害的笑脸与面部肌肉和神经的生理结构联系起来，通过科学术语与日常情感的对比，创造了一种反差的喜剧效果，体现了外表和内部结构之间的戏谑性对比"
        ]
    },
    {
        "original_data": "法国工人开始思考，如果一切都停业了，那我们怎么举行罢工。",
        "annotated_results": [
            "这句话通过幽默的方式指出，罢工的目的是让工作停摆，但如果所有事情已经停业，那么罢工本身就失去了意义。",
            "法国工人经常罢工，罢工意味着拒绝工作，所以他们思考如果没有工作需求，那么如何罢工呢",
            "这里的法国工人是罢工潮的主力军，揭示他们对罢工目的的困惑，反映经济停滞时的矛盾。"
        ]
    },
    {
        "original_data": "出生不久的小牛犊饿死了，原因是空腹不能喝牛奶",
        "annotated_results": [
            "这句话可能是讽刺或夸张表达，指出了某种矛盾，即小牛因为饥饿而需要牛奶，但同时又不能空腹饮用，导致了它的死亡。",
            "在医学上，空腹喝牛奶可能会造成腹泻、恶心等一系列不良反应。这句话展示了荒谬的因果关系，暗示小牛犊因无法喝牛奶而饿死的理由不合理，反映了逻辑上的错误。",
            "刚出生的小牛因饥饿而死亡，小牛死亡的具体原因是小牛在空腹状态下无法消化或吸收牛奶"
        ]
    },
    {
        "original_data": "王警官一生气就用拳头猛锤桌面，破案率居高不下",
        "annotated_results": [
            "这句话以夸张的方式暗示王警官的暴力行为和他的高破案率之间似乎有某种荒谬的因果关系，实际上是讽刺了简单粗暴的执法手段。",
            "这句话把破案的案和桌面的案台锤破巧妙的进行类比，因此有了猛锤桌面就破案率居高不下。",
            "破案率本意为侦破案件的概率，但同时破有破坏之意，案有桌子的意思，因此此处的破案率实际意思为破坏桌子的概率，王警官经常锤坏桌子。"
        ]
    },
    {
        "original_data": "警官当场枪毙了柯南，日本犯罪率下降了一半！",
        "annotated_results": [
            "这是对柯南这个侦探角色的讽刺性夸张表达，暗示如果没有他犯罪也会减少，讽刺了案件的复杂性可能是人为推高的。",
            "柯南经常发现犯罪事件，枪毙了柯南就没人发现新的犯罪事件了",
            "柯南出场往往伴随着各种犯罪事件，这里采用讽刺的手法标明如果没有他的存在可能会更好"
        ]
    }
]


# 定义提示模板，包含In-Context数据
def create_prompt(original_data, in_context_data):
    prompt = "The following are examples of humorous sentence analysis:\n"
    
    # 加入In-Context Learning的示例
    for example in in_context_data:
        prompt += f"Original sentence: {example['original_data']}\n"
        prompt += "Annotated results:\n"
        for i, result in enumerate(example['annotated_results'], 1):
            prompt += f"{i}. {result}\n"
        prompt += "\n"
    
    # 加入待处理的数据
    prompt += f"Now, analyze the following sentence:\nOriginal sentence: {original_data}\nAnnotated results:"
    return prompt

# 调用Qwen API
def call_qwen_api(prompt):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "qwen-7b",
        "prompt": prompt,
        "max_tokens": 100,
    }
    response = requests.post(qwen_api_url, headers=headers, data=json.dumps(payload))
    return response.json()

# 调用GPT-4o-mini API
def call_gpt_4o_mini_api(prompt):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "gpt-4o-mini",
        "prompt": prompt,
        "max_tokens": 100,
    }
    response = requests.post(gpt_4o_mini_url, headers=headers, data=json.dumps(payload))
    return response.json()

import json
from torch.utils.data import DataLoader
from utils.rzbDataset import rzbDataset

# k值和数据集
k = 4
val_dataset = rzbDataset("data", k, mode="val")

# 获取唯一的原始数据
origin = val_dataset.data["original"].drop_duplicates().tolist()

# 批量大小
batch_size = 8  # 可以根据需求调整

# 定义 predict 函数，假设它支持批量输入
def predict_batch(batch_inputs):
    # 这里假设 predict 函数内部已经修改支持批处理
    # 将 batch_inputs 作为列表传递给API，返回批量预测结果
    # 根据API的具体调用进行调整
    batch_results = []
    for input_text in batch_inputs:
        # 调用具体的API进行推理
        # 这里假设你使用的API为 Qwen 或 GPT-4o-mini
        result = call_qwen_api(input_text)  # 或者 call_gpt_4o_mini_api(input_text)
        batch_results.append(result["text"])  # 获取预测的结果文本
    return batch_results

# 构建DataLoader
class TextDataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 使用DataLoader将数据分批加载
data_loader = DataLoader(TextDataset(origin), batch_size=batch_size, shuffle=False)

# 存储结果
result = []

# 批量处理
for batch_idx, batch_inputs in enumerate(data_loader):
    # 批量预测
    predicted_texts = predict_batch(batch_inputs)
    
    # 生成批次结果
    for i, predicted_text in enumerate(predicted_texts):
        subresult = {
            "id": batch_idx * batch_size + i,
            "original": batch_inputs[i],
            "inference": predicted_text
        }
        result.append(subresult)

# 保存结果到文件
with open("batch_inference_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"批量处理完成，结果已保存到 batch_inference_results.json 文件中。")
