import pandas as pd
import numpy as np
import torch
import os
import re

import json

import torch.nn as nn

from sklearn.metrics import f1_score, accuracy_score,roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from nltk import sent_tokenize
from utils.config import tokenizer
from datetime import datetime

def collate_fn_concat(batch):
    """
    Collate function for concatenation/trunction method
    Input: dict
    Output: dict
    """

    # Convert lists of lists into tensor batches
    padded_input_ids_list = [item['input_ids'].clone().detach() for item in batch]  # (num_messages, 512)
    padded_attention_mask_list = [item['attention_mask'].clone().detach() for item in batch]  # (num_messages, 512)
    padded_label_list = [item['labels'].clone().detach() for item in batch]  # Labels are scalars

    # Find the max number of messages (chunks) in this batch
    max_chunks = max(tensor.shape[0] for tensor in padded_input_ids_list)

    # Pad each patient's messages to match max_chunks
    def pad_to_max_chunks(tensor_list, pad_value=0):
        return torch.stack([
            torch.cat([t, torch.full((max_chunks - t.shape[0], t.shape[1]), pad_value, dtype=torch.int64)])
            if t.shape[0] < max_chunks else t
            for t in tensor_list
        ])

    padded_input_ids = pad_to_max_chunks(padded_input_ids_list, pad_value=0)  # Shape: (batch_size, max_chunks, 512)
    padded_attention_mask = pad_to_max_chunks(padded_attention_mask_list, pad_value=0)  # Same shape as input_ids

    # Stack labels (no need for padding)
    padded_labels = torch.stack(padded_label_list)  # Shape: (batch_size,)

    # Final padded batch
    padded_batch = {
        'input_ids': padded_input_ids,  # (batch_size, max_chunks, 512)
        'attention_mask': padded_attention_mask,  # (batch_size, max_chunks, 512)
        'labels': padded_labels  # (batch_size,)
    }
    
    return padded_batch

def compute_metrics_concat(logits, labels,threshold=0.5):
    # Ensure logits and labels are on the CPU and converted to NumPy arrays for metrics computation
    logits = logits.detach().cpu().numpy()  # Detach from the graph and move to CPU
    labels = labels.detach().cpu().numpy()

    # Apply sigmoid to get probabilities for the positive class (class 1)
    probabilities = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    # Get predicted labels from logits (thresholding at 0.5 for binary classification)
    predictions = (probabilities > threshold).astype(int)

    # Compute F1 score, accuracy, ROC AUC, recall, and precision
    f1 = f1_score(labels, predictions, average='weighted')  # Weighted F1 score
    accuracy = accuracy_score(labels, predictions)
    auc_roc = roc_auc_score(labels, probabilities)  # For binary classification, use probabilities of class 1
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted',zero_division=0)

    return {"f1": f1, "accuracy": accuracy, "auroc": auc_roc, "recall": recall, "precision": precision}

def flatten_list(input_list):
    """
    Flattens a ragged list of lists to a list
    Input: list
    Output: list
    """

    return [item for sublist in input_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def aggregate_and_process_pt_with_chunks(enc_list):
    """
    Chunks all of one patient's messages each of their encounters for input to hierarchical method
    Input: list of dict
    Output: list of string
    """

    all_msgs = list(map(lambda d: d['content'], sum(list(map(lambda d: d['pat_messages'], enc_list)), [])))
    all_msg_lens = list(map(lambda d: d['word_count'], sum(list(map(lambda d: d['pat_messages'], enc_list)), [])))
    indices = np.argwhere(np.array(all_msg_lens) > 318)
    if indices.shape[0] == 0:
        return all_msgs
    else:
        for i in np.concatenate(indices):
            all_msgs[i] = chunk(all_msgs[i]) 
        return flatten_list(all_msgs)

def aggregate_and_process_pt_no_chunks(enc_list):
    all_msgs = list(map(lambda d: d['content'], sum(list(map(lambda d: d['pat_messages'], enc_list)), [])))
    return flatten_list(all_msgs)


def chunk(text, chunk_size=318):
    """
    Chunk text to length <= chunk_size
    Input: text: string, chunk_size: int
    Output: list of string
    """

    sentences = sent_tokenize(text)  # Tokenize text into sentences
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(re.findall(r'\S+', sentence))  # Count words in the sentence
        
        # If adding this sentence exceeds chunk size, start a new chunk
        if current_word_count + words_in_sentence > chunk_size:
            chunks.append(" ".join(current_chunk))  # Save the current chunk
            current_chunk = []  # Start a new chunk
            current_word_count = 0  # Reset word count
        
        current_chunk.append(sentence)  # Add sentence to chunk
        current_word_count += words_in_sentence  # Update word count
    
    # Add last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def tokenize_chunks(message_chunks):
    """
    Tokenizes a list of message chunks
    Input: list
    Output: dict
    """

    tokenized = tokenizer(message_chunks, padding='longest', truncation=True, max_length=512, return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"], 
        "attention_mask": tokenized["attention_mask"]
    }

def collate_fn_pooling(batch):
    """
    Collate function for hierarchical method
    Input: dict
    Output: dict
    """

    input_ids_list = [item['input_ids'].clone().detach() for item in batch]  # List of (k, 512) tensors
    attention_mask_list = [item['attention_mask'].clone().detach() for item in batch]  # List of (k, 512) tensors
    label_list = [item['labels'].clone().detach() for item in batch]
    
    # Pad sequences within each chunk to match the max sequence length within the batch
    batch_max_seq_len = max(list(map(lambda x: x.shape[1], input_ids_list))) 
    padded_input_ids_list = [nn.functional.pad(tensor, (0, batch_max_seq_len-tensor.shape[1]), value=0) for tensor in input_ids_list]  # Pad each sequence in a chunk
    padded_attention_mask_list = [nn.functional.pad(tensor, (0, batch_max_seq_len-tensor.shape[1]), value=0) for tensor in attention_mask_list]  # Same for attention masks
    
    # Pad across chunks to match the max number of chunks in the batch
    max_chunks = max(len(item) for item in input_ids_list)  # Find max number of chunks in batch
    padded_input_ids = nn.utils.rnn.pad_sequence(padded_input_ids_list, batch_first=True, padding_value=0)
    padded_attention_mask = nn.utils.rnn.pad_sequence(padded_attention_mask_list, batch_first=True, padding_value=0)
    
    padded_input_ids = nn.functional.pad(padded_input_ids, (0, 0, 0, max_chunks - padded_input_ids.shape[1]), value=0)
    padded_attention_mask = nn.functional.pad(padded_attention_mask, (0, 0, 0, max_chunks - padded_attention_mask.shape[1]), value=0)
    # Final padded batch
    padded_labels = torch.stack(label_list)
    padded_batch = {'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask,'labels':padded_labels}
    return padded_batch


def compute_metrics_pooling(logits, labels, threshold=0.5):
    # Ensure logits and labels are on the CPU and converted to NumPy arrays for metrics computation
    logits = logits.detach().cpu().numpy()  # Detach from the graph and move to CPU
    labels = labels.detach().cpu().numpy()

    # Apply sigmoid to get probabilities for the positive class (class 1)
    #TODO: Make both pooling and concat method functions the same (differ only by one line)
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    # Get predicted labels from logits (thresholding at 0.5 for binary classification)
    predictions = (probabilities > threshold).astype(int)

    # Compute F1 score, accuracy, ROC AUC, recall, and precision
    f1 = f1_score(labels, predictions, average='weighted')  # Weighted F1 score
    accuracy = accuracy_score(labels, predictions)
    auc_roc = roc_auc_score(labels, probabilities)  # For binary classification, use probabilities of class 1
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted',zero_division=0)

    return {"f1": f1, "accuracy": accuracy, "auroc": auc_roc, "recall": recall, "precision": precision}

def extract_before_and_after(pt_id, enc_list, time_option, time_df):
    """
    Extract all the messages before OR after some timestamp for each patient
    Input: pt_id: string, enc_list: list of dict, time_option: string, time_df: dataframe
    Output: list of dict
    """

    if time_option != 'before' and time_option != 'after':
        return None
    before = []
    enc_filtered = list(map(lambda d: 1 if datetime.strptime(d['first_msg_time'],"%Y-%m-%d %H:%M:%S") >= time_df[pt_id] else 0, enc_list))
    for i in range(len(enc_filtered)):
        if time_option == "before" and enc_filtered[i] == 0:
            before.append(enc_list[i])
        elif time_option == "after" and enc_filtered[i] == 1:
            before.append(enc_list[i])
    return before
