import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# Parameters
model_name = "LSTM-seq2seq"
fold_nums = 10
num_epochs = 10
max_length = 128
batch_size = 8
lr = 1e-4
hidden_size = 512  # Hidden size for LSTM
num_layers = 2  # Number of layers for LSTM
device = torch.device('cpu')

# 加载模型和分词器
model_name = "google/mt5-base"  # 假设你使用的是 MT5 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size
embedding_dim = 256  # Embedding dimension for LSTM

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, max_length):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        logits = self.fc(lstm_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding tokens
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

# Load tokenizer and initialize model
model = Seq2SeqLSTM(vocab_size, embedding_dim, hidden_size, num_layers, max_length)

# 加载模型权重
weights_path = '/mnt/lv01/lrf/RuoZhiBa_Translator/runs/LSTM-seq2seq-2024-10-19-18-16-39/weights/best-2024-10-19-18-16-39.pt'
model.load_state_dict(torch.load(weights_path, map_location=device))

def predict(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024,
                       truncation=True, padding="max_length").to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Decode predictions
    logits, _ = outputs  # unpack outputs tuple
    predictions = torch.argmax(logits, dim=-1)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    return decoded_predictions[0]


# Example usage
input_text = "秦始皇因不想学习外语连灭六国，堪称最讨厌外语的学生"
predicted_text = predict(input_text)
print('input:', input_text)
print(f"Predicted text: {predicted_text}")



from utils.rzbDataset import rzbDataset

k=4
val_dataset = rzbDataset(
    "data", k, mode="val")
origin = val_dataset.data["original"].drop_duplicates().tolist()

import json


result = []

# Batch predict
for i in range(len(origin)):
    input_text = origin[i]
    predicted_text = predict(input_text)
    subresult = {
        "id": i,
        "original": input_text,
        "inference": predicted_text
    }
    result.append(subresult)

# Save result to JSON
with open('inference.json', 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, ensure_ascii=False, indent=4)