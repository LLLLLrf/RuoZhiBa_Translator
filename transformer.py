from tqdm import tqdm
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.rzbDataset import rzbDataset
from torch.utils.data import DataLoader
import evaluate

# Parameters
model_name = "google/mt5-base"
fold_nums = 9
num_epochs = 10
max_length = 128
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)


def tokenize_function(sample):
    model_input = tokenizer(
        sample["original"], max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(sample["annotated"], max_length=max_length,
                       truncation=True, padding="max_length")
    model_input["labels"] = labels["input_ids"]
    return model_input


eval_compute_result = []

# K-Fold Cross-Validation
for k in range(fold_nums):
    print(f"Fold {k+1} / {fold_nums} ...")

    train_dataset = rzbDataset(
        "data", k, mode="train")
    val_dataset = rzbDataset(
        "data", k, mode="val")

    # Tokenize dataset
    tokenized_train_datasets = train_dataset.map(
        tokenize_function, batched=True)
    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

    tokenized_train_datasets.set_format(type="torch")
    tokenized_val_datasets.set_format(type="torch")

    # DataLoader
    train_dataloader = DataLoader(
        tokenized_train_datasets, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(tokenized_val_datasets, batch_size=16)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

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
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Evaluate
    metric = evaluate.combine(
        ["rouge", "bleu", "meteor", "bertscore", "accuracy"])
    model.eval()

    for batch in tqdm(eval_dataloader):
        batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions.view(-1),
                         references=batch["labels"].view(-1))

    eval_compute_result.append(metric.compute())
    print(eval_compute_result[-1])

# Load test dataset
test_dataset = rzbDataset("data", mode="test")

# Tokenize dataset
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

tokenized_test_datasets.set_format(type="torch")

# DataLoader
test_dataloader = DataLoader(tokenized_test_datasets, batch_size=16)

# Evaluate
metric = evaluate.combine(
    ["rouge", "bleu", "meteor", "bertscore", "accuracy"])
model.eval()

for batch in tqdm(test_dataloader):
    batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions.view(-1),
                     references=batch["labels"].view(-1))

print("Test result:", metric.compute())

# Save result
with open("result.txt", "w") as f:
    f.write("Evaluation result:\n")
    for i, result in enumerate(eval_compute_result):
        f.write(f"Fold {i+1}:\n")
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
    f.write("Test result:\n")
    for key, value in metric.compute().items():
        f.write(f"{key}: {value}\n")

# Save model
model.save_pretrained(f"weights/{model_name.replace("/", "-")}-{fold_nums}fold")
