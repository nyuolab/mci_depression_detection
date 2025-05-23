#!/bin/bash
mkdir ./outputs

NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i25000-30000 -n1)

PYTHONPATH=. torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT src/mci_training.py "$@"
