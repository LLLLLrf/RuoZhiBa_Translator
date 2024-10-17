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

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Parameters
model_name = "google/mt5-base"
fold_nums = 10
num_epochs = 100
max_length = 128
batch_size = 8
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


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

# K-Fold Cross-Validation
for k in range(1, fold_nums+1):
    print(f"Fold {k} / {fold_nums} ...")

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
    # eval_dataloader = DataLoader(tokenized_val_datasets, batch_size=batch_size)

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
        ["rouge", "bleu", "meteor"])
    dynamic_metric = evaluate.combine(
        ["rouge", "bleu", "meteor"])
    model.eval()

    for batch in tqdm(tokenized_val_datasets):
        print(batch["labels"][1])
        exit()
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
        max_label = -1
        max_score = -1
        for label in batch["labels"]:
            decode_label = tokenizer.batch_decode(
                label, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            score = dynamic_metric.compute(
                predictions=res, references=decode_label)
            if score["meteor"] > max_score:
                max_score = score["meteor"]
                max_label = decode_label
        # print(res)
        # print(max_label)
        metric.add_batch(predictions=res, references=max_label)

    eval_compute_result.append(metric.compute())
    eval_compute_result[-1]["fold"] = k
    print(eval_compute_result[-1])

    # Save model
    if eval_compute_result[-1]["meteor"] > best_result:
        best_result = eval_compute_result[-1]["meteor"]
        model.save_pretrained(
            f"weights/{model_name.replace('/', '-')}-{now}/best-{now}")
    model.save_pretrained(
        f"weights/{model_name.replace('/', '-')}-{now}/last-{now}")
    print(f"Saved model to weights/{model_name.replace('/', '-')}-{now}")

    # Save result
    df = pd.DataFrame(eval_compute_result, rows=["fold"])
    df.to_csv(f"results/{model_name.replace('/', '-')}-{now}.csv")
    print(f"Saved result to results/{model_name.replace('/', '-')}-{now}.csv")

print(f"Best result: {best_result}")

# Save average result
avg_result = {}
for key in eval_compute_result[0].keys():
    if key == "fold":
        continue
    avg_result[key] = sum(
        [result[key] for result in eval_compute_result]) / len(eval_compute_result)
avg_result["fold"] = "average"
print(f"Average result: {avg_result}")
eval_compute_result.append(avg_result)
df = pd.DataFrame(eval_compute_result, rows=["fold"])
df.to_csv(f"results/{model_name.replace('/', '-')}-{now}.csv")
print(
    f"Saved average result to results/{model_name.replace('/', '-')}-{now}.csv")
