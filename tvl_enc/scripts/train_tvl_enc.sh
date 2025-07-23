#!/bin/bash

# Usage: ./script_name.sh gpu_num /path/to/your/data

if [[ $# -eq 2 ]]; then
  GPU_NUM=$1
  DATA_PATH=$2
else
  echo 'gpu_numb=$1 dataset_dir=$2'
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_NUM OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_pretrain.py \
  --batch_size 256 \
  --epochs 200 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  --datasets ssvtp hct \
  --active_modality_names vision tactile text \
  --find_unused_parameters \
  --multi_epochs_dataloader \
  --log_name tvl_vittiny_tactile_encoder_3 \
  --shuffle_text \
  --no_text_prompt \
  --replace_synonyms \
  --num_workers 20 \
  --use_not_contact \
  --tactile_model vit_tiny_patch16_224 \
  --blr 3e-4 \
  --datasets_dir $DATA_PATH
