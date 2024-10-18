import time
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from utils.rzbDataset import rzbDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import evaluate
import pandas as pd
import os
import json

# LSTM Seq2Seq Model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, target_tensor):
        # Encode
        _, (hidden, cell) = self.encoder(input_tensor)
        # Decode
        output, _ = self.decoder(target_tensor, (hidden, cell))
        # Apply final linear layer
        output = self.fc(output)
        return output

# Parameters
input_size = 128  # Embedding size
hidden_size = 256
output_size = 128  # Same as input size
fold_nums = 10
num_epochs = 100
max_length = 128
batch_size = 8
lr = 1e-4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tk_model = "google/mt5-base"

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f">>>>>>>>>>> Start at {now}, device: {device}, model: LSTM-seq2seq, fold_nums: {fold_nums}, num_epochs: {num_epochs}, max_length: {max_length}, batch_size: {batch_size}")
time.sleep(3)

# Save config
if not os.path.exists(f"runs"):
    os.mkdir("runs")
if not os.path.exists(f"runs/LSTM-seq2seq-{now}"):
    os.mkdir(f"runs/LSTM-seq2seq-{now}")

config = {
    "model_name": "LSTM-seq2seq",
    "fold_nums": fold_nums,
    "num_epochs": num_epochs,
    "max_length": max_length,
    "batch_size": batch_size,
    "lr": lr,
    "device": str(device),
    "start_time": now
}

with open(f"runs/LSTM-seq2seq-{now}/config.json", "w") as f:
    json.dump(config, f)

tokenizer = AutoTokenizer.from_pretrained(tk_model)
model = AutoModelForSeq2SeqLM.from_pretrained(tk_model)
model.to(device)

# Load datasets (use your own dataset loader here)
def tokenize_function(sample):
    # Assuming 'original' is input and 'annotated' is target
    input_tensor = tokenizer.encode(sample["original"], return_tensors="pt").squeeze()
    target_tensor = tokenizer.encode(sample["annotated"], return_tensors="pt").squeeze()
    return {"input_tensor": torch.tensor(sample["original"]), "target_tensor": torch.tensor(sample["annotated"])}

eval_compute_result = []
best_result = -1
best_idx = -1
loss_record = []

# Evaluation metric
print('evaluate')
metric = evaluate.combine(["rouge", "bleu", "meteor"])
print('finish evaluate')

# Model
model = Seq2SeqLSTM(input_size, hidden_size, output_size)
model.to(device)

# K-Fold Cross-Validation
for k in range(1, fold_nums + 1):
    print(f"Fold {k} / {fold_nums} ...")
    loss_record.append([])

    train_dataset = rzbDataset("data", k, mode="train")
    val_dataset = rzbDataset("data", k, mode="val")

    train_dataset = Dataset.from_dict(train_dataset.data)
    val_dataset = Dataset.from_dict(val_dataset.data)

    # Tokenize dataset
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

    tokenized_train_datasets.set_format(type="torch")
    tokenized_val_datasets.set_format(type="torch")

    # DataLoader
    train_dataloader = DataLoader(tokenized_train_datasets, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_val_datasets, batch_size=batch_size)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(num_epochs):
        loss_cnt = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {num_epochs}", position=0):
            input_tensor = batch["input_tensor"].to(device)
            target_tensor = batch["target_tensor"].to(device)

            # Forward pass
            outputs = model(input_tensor, target_tensor)

            # Calculate loss
            loss = nn.CrossEntropyLoss()(outputs.view(-1, output_size), target_tensor.view(-1))
            loss_cnt += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        loss_record[-1].append(loss_cnt / len(train_dataloader))
        print(f"Epoch {epoch + 1} loss: {loss_cnt / len(train_dataloader)}")

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader, position=0):
        input_tensor = batch["input_tensor"].to(device)
        target_tensor = batch["target_tensor"].to(device)

        with torch.no_grad():
            outputs = model(input_tensor, target_tensor)

        # Convert to predictions
        predictions = torch.argmax(outputs, dim=-1)
        predictions = predictions.cpu().numpy()
        references = target_tensor.cpu().numpy()

        # Decode and calculate metrics
        metric.add_batch(predictions=predictions, references=references)

    eval_result = metric.compute()
    eval_result["fold"] = k
    eval_compute_result.append(eval_result)

    print(eval_result)

    # Save model
    if eval_result["meteor"] > best_result:
        best_result = eval_result["meteor"]
        best_idx = k
        torch.save(model.state_dict(), f"runs/LSTM-seq2seq-{now}/weights/best.pth")
    torch.save(model.state_dict(), f"runs/LSTM-seq2seq-{now}/weights/last.pth")
    print(f"Saved model to runs/LSTM-seq2seq-{now}/weights")

    # Save result
    df = pd.DataFrame(eval_compute_result)
    df.to_csv(f"runs/LSTM-seq2seq-{now}/result.csv")
    print(f"Saved result to runs/LSTM-seq2seq-{now}/result.csv")

print(f"Best result: {best_result}, fold: {best_idx}")

# Save average result
avg_result = {}
for key in eval_compute_result[0].keys():
    if key == "fold":
        continue
    avg_result[key] = sum([result[key] for result in eval_compute_result]) / len(eval_compute_result)
avg_result["fold"] = "average"
eval_compute_result.append(avg_result)

# Save final results
df = pd.DataFrame(eval_compute_result)
df.to_csv(f"runs/LSTM-seq2seq-{now}/final_results.csv")
df = pd.DataFrame(loss_record)
df.to_csv(f"runs/LSTM-seq2seq-{now}/loss.csv")
print(f"Saved all results to runs/LSTM-seq2seq-{now}")
