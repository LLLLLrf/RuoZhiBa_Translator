import time
import datetime
from tqdm import tqdm
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.rzbDataset import rzbDataset
from torch.utils.data import DataLoader
from datasets import Dataset
import evaluate
import pandas as pd
import os
import json


# Parameters
model_name = "google/mt5-base"
fold_nums = 10
num_epochs = 100
max_length = 128
batch_size = 8
lr = 1e-4
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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
metric = evaluate.combine(
    ["rouge", "bleu", "meteor"])

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
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

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
            outputs = model(**batch)
            loss = outputs.loss
            loss_cnt += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_record[-1].append(loss_cnt / len(train_dataloader))
        print(f"Epoch {epoch+1} loss: {loss_cnt / len(train_dataloader)}")

    model.eval()
    for batch in tqdm(eval_dataloader, position=0):
        labels = batch["labels"]
        batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["input_ids"].to(device)
        }

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        res = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        res = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        metric.add_batch(predictions=res, references=label)

    eval_compute_result.append(metric.compute())
    eval_compute_result[-1]["fold"] = k
    print(eval_compute_result[-1])

    # Save model
    if eval_compute_result[-1]["meteor"] > best_result:
        best_result = eval_compute_result[-1]["meteor"]
        best_idx = k
        model.save_pretrained(
            f"runs/{model_name.replace('/', '-')}-{now}/weights/best-{now}")
    model.save_pretrained(
        f"runs/{model_name.replace('/', '-')}-{now}/weights/last-{now}")
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
