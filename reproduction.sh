#!/usr/bin/env bash

#0 Install and activate environment
echo "Creating project environment"
python3 -m venv mci_detection_environment

source ./mci_detection_environment/bin/activate

pip install -r requirements.txt
echo "Environment creation done!"

#1 Retrieve patient messages
echo "Starting message retrieval"
./retrieve.sh
echo "Message retrieval done!"

#2 Phenotype patient messages
echo "Starting phenotyping retrieval"
./phenotyper.sh
echo "Phenotyping done!"

#3 
#Tokenize messages for task 1 (MCI detection) using concatenation and mental-bert
echo "Starting tokenization"

./tokenization.sh --task_type 1

#Tokenize messages for task 2 (pre post MCI detection) concatenation and mental-bert

./tokenization.sh --task_type 2

#Tokenize messages for task 1 (MCI detection) using hierarchical method and mental-bert

./tokenization.sh --task_type 1 --method 'hierarchical'

#Tokenize messages for task 2 (pre post MCI detection) using trunctation and mental-bert

./tokenization.sh --task_type 2 --method 'hierarchical'
echo "Tokenization done!"

#4 
#Train for task 1 using concatenation method
echo "Starting model trainings"
./training.sh --datapath './data/mental/mental-bert-base-uncased_tokenized_data_task_1_concat'

#Train for task 2 using concatenation method

./training.sh --datapath './data/mental/mental-bert-base-uncased_tokenized_data_task_2_concat'

#Train for task 1 using hierarchical method

./training.sh --datapath './data/mental/mental-bert-base-uncased_tokenized_data_task_1_hierarchical' --method 'hierarchical'

#Train for task 2 using hierarchical method

./training.sh --datapath './data/mental/mental-bert-base-uncased_tokenized_data_task_2_hierarchical' --method 'hierarchical'
echo "Training done!"
