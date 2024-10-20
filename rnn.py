import time
import datetime
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from utils.rzbDataset import rzbDataset

import evaluate
import pandas as pd
import os
import json
import torch.nn as nn


# Parameters
model_name = "RNN-seq2seq"
fold_nums = 10
num_epochs = 50
max_length = 128
batch_size = 8
lr = 1e-4
hidden_size = 512  # Hidden size for RNN
num_layers = 2  # Number of layers for RNN
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f">>>>>>>>>>> Start at {now}, device: {device}, model: {model_name}, fold_nums: {fold_nums}, num_epochs: {num_epochs}, max_length: {max_length}, batch_size: {batch_size}")
time.sleep(3)

# Save config
if not os.path.exists(f"runs"):
    os.mkdir("runs")
if not os.path.exists(f"runs/{model_name.replace('/', '-')}-{now}"):
    os.mkdir(f"runs/{model_name.replace('/', '-')}-{now}")

config = {
    "model_name": model_name,
    "fold_nums": fold_nums,
    "num_epochs": num_epochs,
    "max_length": max_length,
    "batch_size": batch_size,
    "lr": lr,
    "device": str(device),
    "start_time": now
}

with open(f"runs/{model_name.replace('/', '-')}-{now}/config.json", "w") as f:
    json.dump(config, f)

# RNN-based Seq2Seq Model
class Seq2SeqRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, max_length):
        super(Seq2SeqRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        rnn_output, _ = self.rnn(embedded)
        logits = self.fc(rnn_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding tokens
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

# Load tokenizer and initialize model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")  # still use the MT5 tokenizer
vocab_size = tokenizer.vocab_size
embedding_dim = 256  # Embedding dimension for RNN

model = Seq2SeqRNN(vocab_size, embedding_dim, hidden_size, num_layers, max_length)
model.to(device)

def tokenize_function(sample):
    model_input = tokenizer(
        sample["original"], max_length=max_length, truncation=True, padding="max_length")
    if type(sample["annotated"]) == str:
        labels = tokenizer(sample["annotated"], max_length=max_length,
                           truncation=True, padding="max_length")
        model_input["labels"] = labels["input_ids"]
    else:
        labels = [tokenizer(annotated, max_length=max_length, truncation=True,
                            padding="max_length") for annotated in sample["annotated"]]
        model_input["labels"] = [label["input_ids"] for label in labels]
    return model_input


eval_compute_result = []
best_result = -1
best_idx = -1
loss_record = []

# Evaluate
print('evaluate')
metric = evaluate.combine(
    ["rouge", "bleu", "meteor"])
print('finish evaluate')

# K-Fold Cross-Validation
for k in range(1, fold_nums+1):
    print(f"Fold {k} / {fold_nums} ...")

    loss_record.append([])

    train_dataset = rzbDataset(
        "data", k, mode="train")
    val_dataset = rzbDataset(
        "data", k, mode="val")

    train_dataset = Dataset.from_dict(train_dataset.data)
    val_dataset = Dataset.from_dict(val_dataset.data)

    # Tokenize dataset
    tokenized_train_datasets = train_dataset.map(
        tokenize_function, batched=True)
    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

    tokenized_train_datasets.set_format(type="torch")
    tokenized_val_datasets.set_format(type="torch")

    # DataLoader
    train_dataloader = DataLoader(
        tokenized_train_datasets, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_val_datasets, batch_size=batch_size)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=num_training_steps)

    # Train
    model.train()
    for epoch in range(num_epochs):
        loss_cnt = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} / {num_epochs}", position=0):
            batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            logits, loss = model(batch["input_ids"], labels=batch["labels"])
            loss_cnt += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_record[-1].append(loss_cnt / len(train_dataloader))
        print(f"Epoch {epoch+1} loss: {loss_cnt / len(train_dataloader)}")

    # Evaluation step (same as before)
    model.eval()
    for batch in eval_dataloader:
        labels = batch["labels"]
        batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        with torch.no_grad():
            logits, _ = model(batch["input_ids"])

        predictions = torch.argmax(logits, dim=-1)
        res = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        metric.add_batch(predictions=res, references=label)

    eval_compute_result.append(metric.compute())
    eval_compute_result[-1]["fold"] = k
    print(eval_compute_result[-1])

    # Save model
    if eval_compute_result[-1]["meteor"] > best_result:
        best_result = eval_compute_result[-1]["meteor"]
        best_idx = k
        save_dir = f"runs/{model_name.replace('/', '-')}-{now}/weights"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(),
                f"runs/{model_name.replace('/', '-')}-{now}/weights/best-{now}.pt")
    torch.save(model.state_dict(),
               f"runs/{model_name.replace('/', '-')}-{now}/weights/last-{now}.pt")
    print(f"Saved model to runs/{model_name.replace('/', '-')}-{now}/weights")

    # Save result
    df = pd.DataFrame(eval_compute_result)
    df.to_csv(f"runs/{model_name.replace('/', '-')}-{now}/result.csv")
    print(
        f"Saved result to runs/{model_name.replace('/', '-')}-{now}/result.csv")

print(f"Best result: {best_result}, fold: {best_idx}")

# Save average result
avg_result = {}
for key in eval_compute_result[0].keys():
    if key == "fold":
        continue
    if type(eval_compute_result[0][key]) == list:
        avg_result[key] = []
        for i in range(len(eval_compute_result[0][key])):
            avg_result[key].append(sum(
                [result[key][i] for result in eval_compute_result]) / len(eval_compute_result))
    else:
        avg_result[key] = sum(
            [result[key] for result in eval_compute_result]) / len(eval_compute_result)
avg_result["fold"] = "average"
print(f"Average result: {avg_result}")
eval_compute_result.append(avg_result)
df = pd.DataFrame(eval_compute_result)
df.to_csv(f"runs/{model_name.replace('/', '-')}-{now}/result.csv")
df = pd.DataFrame(loss_record)
df.to_csv(f"runs/{model_name.replace('/', '-')}-{now}/loss.csv")
print(f"Saved to runs/{model_name.replace('/', '-')}-{now}")

