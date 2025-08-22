from utils.model_utils import merge_labels, truncate_tokens
from utils.models import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.metrics import recall_score

import pandas as pd

import json
import argparse
import sys
import os
import gc
import math

from sklearn.metrics import confusion_matrix


def soft_cross_entropy(preds, soft_targets):
    log_probs = F.log_softmax(preds, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()  # Weight with noise


# Helper functions for setting up distributed training
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_dpp():
    dist.destroy_process_group()


# Helper function to parse arguments
def parse_args():
    # 1. PARSE ARGS
    parser = argparse.ArgumentParser(
        description="Arguments: datapath (required)"
    )
    parser.add_argument(
        "--datapath", type=str, required=True, help="Path of tokenized dataset"
    )
    args = parser.parse_args()

    return args


def gather_tensor(tensor):
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def main():
    # Run ddp setup
    local_rank = setup_ddp()
    rank = dist.get_rank()
    args = parse_args()
    # Settings
    MODEL_NAME = "answerdotai/ModernBERT-base"
    EXPERIMENT_NAME = "ABLATION_NO_REGEX"
    BATCH_SIZE = 256
    NUM_EPOCHS = 3
    ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    # Load tokenized data
    train_dataset = Dataset.from_parquet(args.datapath, cache_dir='scratch')
    train_dataset = train_dataset.map(merge_labels)
    train_dataset = train_dataset.remove_columns(column_names=["content", "__index_level_0__"])
    train_dataset = train_dataset.map(truncate_tokens)
    train_dataset = train_dataset.remove_columns(column_names=["label_0", "label_1"])

    test_dataset = Dataset.from_parquet('test_snorkel.parquet' cache_dir='hf_datasets_cache')
    test_dataset = test_dataset.remove_columns(column_names=["content"])
    test_dataset = test_dataset.map(truncate_tokens)
    test_dataset = test_dataset.map(lambda x: {"labels": [0.0,1.0]})

    reserved_test = test_dataset.select(range(8000))  
    remaining_test = test_dataset.select(range(8000, len(test_dataset)))

    if rank == 0:
        print(reserved_test.column_names)
        print(remaining_test.column_names)
        print(test_dataset.column_names)
        print(train_dataset.column_names)

    dataset = concatenate_datasets([train_dataset, remaining_test])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors="pt"
    )

    train_sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
            reserved_test,
            batch_size=BATCH_SIZE,
            collate_fn=data_collator,
    )
    # Device setup
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if rank == 0:
        EXPERIMENT_NAME = args.datapath.split("/")[-1].split(".")[0]
        output_dir = f"snorkel_ablation_results/output_{MODEL_NAME}_{EXPERIMENT_NAME}"
        logging_dir = f"snorkel_ablation_results/logs_{MODEL_NAME}_{EXPERIMENT_NAME}"
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create output directory if it doesn't exist
        os.makedirs(
            logging_dir, exist_ok=True
        )  # Create logging directory if it doesn't exist

    train_losses = []
    # Model, optimizer, scaler
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True
    )
    model._set_static_graph()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler("cuda", enabled=True)
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in tqdm(
            enumerate(train_loader, 1), total=len(train_loader)
        ):
            # Move inputs & labels to device
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            soft_labels = batch["labels"].to(device)  # shape (B,2)

            with torch.autocast(device_type="cuda"):
                logits = model(**inputs).logits  # (B,2)
                loss = soft_cross_entropy(logits, soft_labels)

            # Backpropagate
            scaler.scale(loss).backward()

            # Step optimizer
            if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

        train_losses.append(running_loss / step)
        train_loss_tensor = torch.tensor([running_loss / step], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss_avg = train_loss_tensor.item() / dist.get_world_size()
        if rank == 0:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_preds = []
            test_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    inputs = {
                            "input_ids": batch["input_ids"].to(device),
                            "attention_mask": batch["attention_mask"].to(device),
                            }

                    labels = batch["labels"][:,1].long().to(device)
                    with torch.autocast(device_type="cuda"):
                        outputs = model(**inputs).logits
                        loss = F.cross_entropy(outputs, labels)
                    test_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)

                    test_preds.extend(preds.cpu().tolist())
                    test_labels.extend(labels.cpu().tolist())

            test_loss_avg = test_loss / len(test_loader)
            test_accuracy = test_correct / test_total
            test_recall = recall_score(test_labels, test_preds, average="binary")

            print("Test Results:")
            print(f"Loss: {test_loss_avg:.4f}")
            print(f"Accuracy: {test_accuracy:.2%}")
            print(f"Recall: {test_recall:.2%}\n")

            training_stats = {
                "train_losses": train_losses,
                "test_loss": test_loss_avg,
                "test_accuracy": test_accuracy,
                "test_recall": test_recall,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
            }

            with open(f"{logging_dir}/training_stats.json", "w") as f:
                json.dump(training_stats, f, indent=2)
                print(
                    f"Updated training stats saved \
                            to '{logging_dir}/training_stats.json'."
                )

            state_dict = model.module.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "best_model.pth"))
            print("Training complete.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
