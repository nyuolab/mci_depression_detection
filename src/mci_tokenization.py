import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import re
from datetime import datetime
import pandas as pd


#TODO: Put your data paths here
pt_msg_path = 
mci_phenotype_path = 
demographics_path = 

if not name == "__main__":
    print("mci_tokenization.py is not a module!")

if name == "__main__":
    #1. PARSE ARGS
    #Get task type and method type (default is truncation)
    parser = argparse.ArgumentParser(description="Script requires at least one argument")
    parser.add_argument('task_type', help='Task type argument (required)')
    parser.add_argument('method', help="Method type (optional): hierarchical \nDefault is truncation")
    parser.add_argument("model_name", help="Model name from huggingface. \n Default is mental-bert-base-uncased")
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.error("No arguments provided. Must specifiy task type to proceed")

    #Load patient message data
    df = pd.read_json(pt_msg_path)
    df = df[df.pat_to_doctor_msgs >= 1]

    #2. PHENOTYPE BASE ON TASK
    mci_cohort = pd.read_csv(mci_phenotype_path)
    merged_demos = pd.read_csv(demographics_path)
    mci_set = merged_demos[merged_demos.pat_owner_id.isin(mci_cohort.pat_owner_id)][merged_demos.age >= 50]['pat_owner_id']
    df['label'] = 0
    df['label'] = df.pat_owner_id.isin(mci_set).astype(int)

    #TASK 1: MCI vs NON-MCI
    #Balance dataset for training (TODO: balance only for training then test on regular distribution)
    if(args.task_type == 1):
        majority_class = df[df.label == 0]
        minority_class = df[df.label == 1]
        majority_class_undersampled = resample(majority_class,
                                               replace=False,
                                               n_samples=len(minority_class),
                                               random_state=42)
        balanced_data = pd.concat([majority_class_undersampled, minority_class])

        if(args.method == "hierarchical"):
            balanced_data['all_pt_messages'] = balanced_data['refined_pat_encounters'].progress_apply(lambda x: aggregate_and_process_pt_with_chunks(x))
        else:
            balanced_data['all_pt_messages'] = balanced_data['refined_pat_encounters'].progress_apply(lambda x: aggregate_and_process_pt_no_chunks(x))

        balanced_data.drop(['refined_pat_encounters'],axis=1,inplace=True)
        labeled_df = balanced_data

    #TASK 2: MCI ONLY, Pre vs Post
    if(args.task_type == 2)
        task2 = df[df.label == 1]
        mci_times = mci_cohort.drop('Unnamed: 0', axis=1).set_index('pat_owner_id').to_dict()
        mci_times = mci_times['earliest_sign']
        mci_times = {k: datetime.strptime(v, "%Y-%m-%d") for k, v in mci_times.items()}

        task2['pre_mci'] = task2.apply(lambda row: extract(row['pat_owner_id'],row['refined_pat_encounters'],'before', mci_times),axis=1)
        task2['post_mci'] = task2.apply(lambda row: extract(row['pat_owner_id'],row['refined_pat_encounters'],'after', mci_times),axis=1)
        df_pre_mci = task2[task2.pre_mci.apply(len) > 0][['pat_owner_id','pre_mci']]
        df_pre_mci['label'] = 0
        df_post_mci = task2[task2.post_mci.apply(len) > 0][['pat_owner_id','post_mci']]
        df_post_mci['label'] = 1

        if(args.method == "hierarchical"):
            df_pre_mci['all_pt_messages'] = task2['pre_mci'].apply(lambda x: aggregate_and_process_pt_with_chunks(x))
            df_post_mci['all_pt_messages'] = task2['post_mci'].apply(lambda x: aggregate_and_process_pt_with_chunks(x))
        else:
            df_pre_mci['all_pt_messages'] = task2['pre_mci'].apply(lambda x: aggregate_and_process_pt_no_chunks(x))
            df_post_mci['all_pt_messages'] = task2['post_mci'].apply(lambda x: aggregate_and_process_pt_no_chunks(x))

        df_pre_and_post_labeled = pd.concat([df_pre_mci.drop(['pre_mci','pat_owner_id'],axis=1), df_post_mci.drop(['post_mci','pat_owner_id'],axis=1)])

        df_pre_and_post_labeled.reset_index(inplace=True)
        labeled_df = df_pre_and_post_labeled
    #3. TOKENIZATION

    #Choose model and load tokenizer
    if(args.model_name):
        MODEL_NAME = args.model_name
    else:
        MODEL_NAME = "mental/mental-bert-base-uncased"

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

    if(args.method == "hierarchical"):
        labeled_df['tokenized'] = labeled_df['all_pt_messages'].apply(tokenize_chunks)
        dataset_dict = {
                'input_ids': labeled_df['tokenized'].apply(lambda x: x['input_ids'].tolist()).tolist()
                'attention_mask': labeled_df['tokenized'].apply(lambda x: x['attention_mask'].tolist()).tolist(),
                'labels': labeled_df['label'].tolist()
        }
        dataset = Dataset.from_dict(dataset_dict).with_format('torch')
    else:
        texts = labeled_df['all_pt_messages']
        labels = labeled_df['label']

        tokenized_texts = []
        attention_masks = []

        for text_list in tqdm(texts):
            combined_text = ' [SEP] '.join(text_list)
            inputs = tokenizer.encode_plus(
                    combined_text, add_special_tokens=True, max_length=max_len, 
                    padding='max_length', truncation=True, return_tensors='pt'
            )
        tokenized_texts.append(inputs['input_ids'].squeeze().tolist())
        attention_masks.append(inputs['attention_mask'].squeeze().tolist())
        labels = list(labels)
        dataset_dict = {
                'input_ids': tokenized_texts,
                'attention_mask': attention_masks,
                'labels': labels
        }
        dataset = Dataset.from_dict(dataset_dict).with_format('torch')

    #4. SAVE TOKENIZED DATA
    switch = {
            1: 'task_1',
            2: 'task_2',
            }
    if args.method == 'hierarchical':
        save_path = f"./{MODEL_NAME}_tokenized_data_{switch[args.task_type]}_hierarchical"
    else:
        save_path = f"./{MODEL_NAME}_tokenized_data_{switch[args.task_type]}_concat"

    dataset.save_to_disk(save_path)
