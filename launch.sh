#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate 
# Run training
python train.py \
    --dataroot data_pipe/export_underwater2/dataset_20250921_093929 \
    --dataset_mode custom \
    --yaml_path data_pipe/export_underwater2/dataset_20250921_093929/dataset.yaml \
    --model cwr \
    --name cwr_test_all
    --continue_train 
