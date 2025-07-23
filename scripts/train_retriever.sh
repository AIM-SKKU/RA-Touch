# !/bin/bash

gpu_num=$1
vit_type=${2:-"base"}
epochs=${3:-60}
batch_size=${4:-256}
port=${5:-23500}

nproc_per_node=$(echo $gpu_num | awk -F',' '{print NF}')

output_dir="./output"
log_name="${vit_type}_retriever_epochs${epochs}_batch${batch_size}_GPU_${nproc_per_node}"

if [ $vit_type == "tiny" ]; then
    tactile_model="vit_tiny_patch16_224"
    encoder_ckpt_path="./weights/tvl_enc_vittiny.pth"
elif [ $vit_type == "small" ]; then
    tactile_model="vit_small_patch16_224"
    encoder_ckpt_path="./weights/tvl_enc_vits.pth"
elif [ $vit_type == "base" ]; then
    tactile_model="vit_base_patch16_224"
    encoder_ckpt_path="./weights/tvl_enc_vitb.pth"
else
    echo "Invalid vit type"
    exit 1
fi

echo ************************************************************
echo log_name: $log_name
echo epochs: $epochs
echo batch_size: $batch_size
echo tactile_model: $tactile_model
echo ************************************************************

CUDA_VISIBLE_DEVICES=$gpu_num python -u -m torch.distributed.launch \
    --master_port=$port --nproc_per_node=$nproc_per_node \
    --use_env ra_touch/train_retriever.py \
    --data_config configs/finetune-data-config.yaml \
    --batch_size $batch_size \
    --epochs $epochs \
    --warmup_epochs 10 \
    --blr 3e-4 \
    --weight_decay 0.02 \
    --output_dir $output_dir  \
    --checkpoint_path $encoder_ckpt_path \
    --tactile_model $tactile_model \
    --log_name $log_name \
    --crop_tacvis \
    --gpu $gpu_num
