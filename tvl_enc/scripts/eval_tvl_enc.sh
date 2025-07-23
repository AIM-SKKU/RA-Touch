#!/bin/bash
#usage: bash eval_tvl_enc.sh gpu_num /path/to/your/checkpoint /path/to/your/dataset

if [[ $# -eq 3 ]] ; then
    GPU_NUM=$1
    CHECKPOINT_PATH=$2
    DATA_PATH=$3
else
    echo 'gpu_num=$1 ckpt_path=$2 dataset_dir=$3'
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_NUM python -m tools.visualize_affinity \
    --checkpoint_path $CHECKPOINT_PATH \
    --visualize_test \
    --active_modality_names tactile text \
    --tactile_model vit_tiny_patch16_224 \
    --no_text_prompt \
    --datasets ssvtp hct \
    --seed 42 \
    --not_visualize \
    --evaluate_all \
    --similarity_thres 0.6356450319 0.8591097295 0.8927201033 0.9208499491 \
    --datasets_dir $DATA_PATH
