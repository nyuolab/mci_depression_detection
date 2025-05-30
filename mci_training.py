from utils.model_utils import *
from utils.models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

import pandas as pd

import json
import argparse
import sys
import os
import gc
import math

from sklearn.metrics import confusion_matrix

#Helper functions for setting up distributed training
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_dpp():
    dist.destroy_process_group()

#Helper function to parse arguments
def parse_args():
    #1. PARSE ARGS
    parser = argparse.ArgumentParser(description="Arguments: datapath (required), model_name (optional), method (optional)")
    parser.add_argument("--datapath", type=str, required=True, help="Path of tokenized dataset")
    parser.add_argument("--model_name", type=str, default="mental/mental-bert-base-uncased",help="Name of model from huggingface")
    parser.add_argument("--method", type=str, default='concat',help="Method: hierarchical. Default is concat")
    args = parser.parse_args()

    return args

def gather_tensor(tensor):
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)

def main(): 
    #Run ddp setup
    local_rank = setup_ddp()
    rank = dist.get_rank()
    args = parse_args()

    #For convinience ...
    MODEL_NAME = args.model_name

    #Load tokenized data
    dataset = Dataset.load_from_disk(args.datapath)
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
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.method == 'hierarchical':
        collate_fn = collate_fn_pooling
        model = BERTGroupClassifier(model_name=MODEL_NAME, num_labels=2)
        # Settings
        BATCH_SIZE = 4
        NUM_EPOCHS = 3
        ACCUMULATION_STEPS = 16
        LEARNING_RATE = 2e-5
        WEIGHT_DECAY = 0.01

    else:
        collate_fn = collate_fn_concat
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
                # Settings
        BATCH_SIZE = 32
        NUM_EPOCHS = 3
        ACCUMULATION_STEPS = 2
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 0.01


    focal_loss = FocalLoss(alpha=0.95, gamma=2.0)
    THRESHOLD = 0.2

    # DataLoaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True);
    val_sampler = DistributedSampler(val_dataset, shuffle=True);
    test_sampler = DistributedSampler(test_dataset, shuffle=False);
    
    train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_fn,
                sampler=train_sampler
                            )

    val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_fn,
                sampler=val_sampler
                            )

    test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_fn,
                sampler=test_sampler
                            )
    # Specify your directories
    if rank == 0:
        EXPERIMENT_NAME = args.datapath.split("/")[-1].split('.')[0]
        output_dir = f"./outputs/output_{MODEL_NAME}_{EXPERIMENT_NAME}"
        logging_dir = f"./outputs/logs_{MODEL_NAME}_{EXPERIMENT_NAME}"
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        os.makedirs(logging_dir, exist_ok=True)  # Create logging directory if it doesn't exist

    train_losses = []
    val_losses = []

    # Model, optimizer, scaler
    model.to(device) 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model._set_static_graph()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda',enabled=True)
    best_f1 = 0.0
    metrics_history = []
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast('cuda',enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = focal_loss(logits,labels) if args.method == 'concat' else outputs.loss
                loss = loss.mean()

            # Backpropagate
            scaler.scale(loss).backward()

            # Step optimizer
            if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                running_loss += loss.item()

        train_losses.append(running_loss / step)
        train_loss_tensor = torch.tensor([running_loss/step],device=device)
        dist.all_reduce(train_loss_tensor,op=dist.ReduceOp.SUM)
        train_loss_avg = train_loss_tensor.item()/dist.get_world_size()
        if rank == 0:
            train_losses.append(train_loss_avg)

        # Validation
        model.eval()
        val_logits_list, val_labels_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for step,batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with autocast('cuda',enabled=True):
                    outputs= model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss = focal_loss(logits,labels) if args.method == 'concat' else outputs.loss
                    loss = loss.mean()
                val_loss += loss.item()
                val_logits_list.append(logits.detach())
                val_labels_list.append(labels.detach())
            val_logits = torch.cat(val_logits_list)
            val_logits = gather_tensor(val_logits)
            val_labels = torch.cat(val_labels_list)
            val_labels = gather_tensor(val_labels)
            if rank == 0:
                val_metrics = compute_metrics(val_logits, val_labels, THRESHOLD)
                metrics_history.append(val_metrics)
                print(f"Epoch {epoch} {val_loss} Validation Metrics: {val_metrics}")

                # Save best model
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    state_dict = model.module.state_dict() 
                    torch.save(state_dict, os.path.join(output_dir, 'best_model.pth'))
                    print(f"Saved best model (F1={best_f1:.4f})")
                val_losses.append(val_loss / step)
        print("Training complete.")

    # Test loop
    if rank == 0:
        print("\nRunning test set evaluation...")
        if args.method == 'hierarchical':
            best_model = BERTGroupClassifier(model_name=MODEL_NAME, num_labels=2).to(device)
        else:
            best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        best_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        best_model = nn.DataParallel(best_model)
        best_model.eval()
        test_logits_list, test_labels_list = [], []
        with torch.no_grad():
            for step,batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with autocast('cuda',enabled=True):
                    outputs = best_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                test_logits_list.append(logits.detach())
                test_labels_list.append(labels.detach())
            test_logits = torch.cat(test_logits_list)
            test_labels = torch.cat(test_labels_list)
            test_metrics = compute_metrics(test_logits, test_labels, THRESHOLD)
            probs = torch.softmax(test_logits, dim=1)[:, 1]
            preds = (probs > THRESHOLD).int()
            cm = confusion_matrix(test_labels.cpu(), preds.cpu())
            print(f"Test Metrics: {test_metrics}")
            training_stats = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_metrics": metrics_history,
                "test_metrics": test_metrics,
                "learning_rate": LEARNING_RATE,
                "batch_size":BATCH_SIZE
        }

        with open(f'{logging_dir}/training_stats.json', 'w') as f:
            json.dump(training_stats, f)
            print(f"Training statistics saved to '{logging_dir}/training_stats.json'.")
        print(args.method)
        print(cm)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
