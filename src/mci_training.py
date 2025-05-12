from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import argparse
import sys
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
import os
import gc
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.model_utils import *

if name == "__main__":
    #1. PARSE ARGS
    parser = argparse.ArgumentParser(description="Arguments: datapath (required), model_name (required), method (optional)")
    parser.add_argument("datapath", help="Path of tokenized dataset")
    parser.add_argument("model_name", help="Name of model from huggingface")
    parser.add_argument("method", help="Method: hierarchical. Default is concat")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.error("Required arguments not provided")

    #For convinience ...
    MODEL_NAME = model_name

    #Load tokenized data
    dataset = Dataset.load_from_disk(datapath)
    dataset.set_format(type='torch')

    dataset = dataset.class_encode_column('labels')
    train_test = dataset.train_test_split(test_size=0.2, seed=42,stratify_by_column='labels')

    # Now, split the training set into 90% train and 10% validation (from the 80% training data)
    train_val = train_test['test'].train_test_split(test_size=0.5, seed=42,stratify_by_column='labels')  # 10% of 80% for validation

    # Now you have: train, validation, and test splits
    train_dataset = train_test['train']
    val_dataset = train_val['test']
    test_dataset = train_val['train']

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == 'hierarchical':
        collate_fn = collate_fn_pooling
        compute_metrics = compute_metrics_pooling
        model = BERTGroupClassifier(model_name=MODEL_NAME, num_labels=1).to(device)
        model.bert.gradient_checkpointing_enable()  # Enable gradient checkpointing
        for param in model.bert.parameters():
            param.requires_grad = False
        # Settings
        BATCH_SIZE = 2
        NUM_EPOCHS = 3
        ACCUMULATION_STEPS = 16
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 0.01

    else:
        collate_fn = collate_fn_concat
        compute_metrics = compute_metrics_concat
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        # Settings
        BATCH_SIZE = 64
        NUM_EPOCHS = 3
        ACCUMULATION_STEPS = 1
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 0.01


    # DataLoaders
    train_loader = DataLoader(
                train_dataset,
                    batch_size=BATCH_SIZE,
                        collate_fn=collate_fn,
                            shuffle=True
                            )
    val_loader = DataLoader(
                val_dataset,
                    batch_size=BATCH_SIZE,
                        collate_fn=collate_fn,
                            shuffle=False
                            )
    test_loader = DataLoader(
                test_dataset,
                    batch_size=BATCH_SIZE,
                        collate_fn=collate_fn,
                            shuffle=False
                            )
    # Specify your directories
    
    EXPERIMENT_NAME = datapath.split("/")[-1].split('.')[0]
    output_dir = f"./output_{model_name}_{EXPERIMENT_NAME}"
    logging_dir = f"./logs_{model_name}_{EXPERIMENT_NAME}"

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    os.makedirs(logging_dir, exist_ok=True)  # Create logging directory if it doesn't exist
    train_losses = []
    val_losses = []

    # Model, optimizer, scaler
    model = nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda',enabled=True)
    best_f1 = 0.0
    metrics_history = []
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        all_preds, all_labels = [], []
        steps_per_update = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
        pbar = tqdm(total=steps_per_update, desc=f"Epoch {epoch}", dynamic_ncols=True,position=0,leave=False)
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast('cuda',enabled=True):
                loss, logits = model(input_ids=input_ids,
                attention_mask=attention_mask,
            labels=labels)
            loss = loss.mean()

            # Backpropagate
            scaler.scale(loss).backward()

            # Gather predictions
            probs = torch.sigmoid(logits.detach())
            batch_preds = (probs > 0.5).long().squeeze(-1).cpu().numpy()

            # Step optimizer
            if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=running_loss / pbar.n)
        train_losses.append(running_loss / step)
        pbar.close()

        # Validation
        model.eval()
        val_logits_list, val_labels_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for step,batch in tqdm(enumerate(val_loader), desc="  Val", leave=False,position=0):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with autocast('cuda',enabled=True):
                    loss, logits = model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                    loss = loss.mean()
                val_loss += loss.item()
                val_logits_list.append(logits.detach())
                val_labels_list.append(labels.detach())
            val_logits = torch.cat(val_logits_list)
            val_labels = torch.cat(val_labels_list)
            val_metrics = compute_metrics(val_logits, val_labels)
            metrics_history.append(val_metrics)
            print(f"Epoch {epoch} {val_loss} Validation Metrics: {val_metrics}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, 'best_model.pth'))
                print(f"Saved best model (F1={best_f1:.4f})")
            val_losses.append(val_losses / step)
        print("Training complete.")

    # Test loop
    print("\nRunning test set evaluation...")
    if args.method == 'hierarchical':
        best_model = BERTGroupClassifier(model_name=MODEL_NAME, num_labels=1).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    best_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    best_model = nn.DataParallel(best_model)
    best_model.eval()
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for step,batch in tqdm(enumerate(test_loader), desc="  Test", leave=False,position=0):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast('cuda',enabled=True):
                _, logits = best_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                        labels=labels)
            test_logits_list.append(logits.detach())
            test_labels_list.append(labels.detach())
        test_logits = torch.cat(test_logits_list)
        test_labels = torch.cat(test_labels_list)
        test_metrics = compute_metrics(test_logits, test_labels)
        print(f"Test Metrics: {test_metrics}")
        training_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics_history,
            "test_metrics": test_metrics,
            "learning_rate": LEARNING_RATE,
            "batch_size":BATCH_SIZE
    }

    with open(f'{logging_dir}/training_stats.json', 'w') as f:
        json.dump(training_stats, f)
        print(f"Training statistics saved to '{logging_dir}/training_stats.json'.")
