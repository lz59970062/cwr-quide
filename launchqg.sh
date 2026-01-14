#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=4
source .venv/bin/activate 

# 将 uwiqf 加入 python 路径
export PYTHONPATH=/home/lizhi_2024/program/udw_compare/uwiqf:$PYTHONPATH
export PYTHONPATH=../uwiqf:$PYTHONPATH
# Run training
# qg_modelconfig="../uwiqf/checkpoints_uwiqf_pairsselfdefect9/basic_training_configselfdefect9.yaml"
# qg_modelcheckpoint="../uwiqf/checkpoints_uwiqf_pairsselfdefect9/checkpoint_epoch_19"
qg_modelconfig="../udw-enhance/checkpoints_uwiqf_pairsselfdefect9/basic_training_configselfdefect9.yaml"  
qg_modelcheckpoint="../udw-enhance/checkpoints_uwiqf_pairsselfdefect9/checkpoint_epoch_19"  
lambda_qg=5

python traincwr.py \
    --dataroot data_pipe/export_underwater2/dataset_20250921_093929 \
    --dataset_mode custom \
    --yaml_path data_pipe/export_underwater2/dataset_20250921_093929/dataset.yaml \
    --model cwr \
    --name cwr_test_qg \
    --lambda_QG $lambda_qg \
    --qg_modelconfig $qg_modelconfig \
    --qg_modelcheckpoint $qg_modelcheckpoint \
    --display_id 0 \
    --batch_size 8