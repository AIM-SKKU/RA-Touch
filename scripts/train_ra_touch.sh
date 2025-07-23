# !/bin/bash
gpu_num=$1
vit_type=$2
retriever_weight=$3
topk=${4:-5}
external_dataset=${5:-"imgnet_t_150k"}
retrieval_method=${6:-"txt2txt"}
port=${7:-1113}

llama_path="./llama-2"
output_dir="./output"
nproc_per_node=$(echo $gpu_num | awk -F',' '{print NF}')
log_name="ra_touch-${vit_type}_${external_dataset}-topK${topk}"

if [ $vit_type == "tiny" ]; then
    tactile_model="vit_tiny_patch16_224"
    pretrained_path="./weights/tvl_llama_vittiny.pth"
    encoder_ckpt_path="./weights/tvl_enc_vittiny.pth"
elif [ $vit_type == "small" ]; then
    tactile_model="vit_small_patch16_224"
    pretrained_path="./weights/tvl_llama_vits.pth"
    encoder_ckpt_path="./weights/tvl_enc_vits.pth"
elif [ $vit_type == "base" ]; then
    tactile_model="vit_base_patch16_224"
    pretrained_path="./weights/tvl_llama_vitb.pth"
    encoder_ckpt_path="./weights/tvl_enc_vitb.pth"
fi

if [ $external_dataset == "imgnet_t_10k" ]; then
    embedding_path="./data/embeddings/imagenet_t_10k_embeddings.npz"
elif [ $external_dataset == "imgnet_t_50k" ]; then
    embedding_path="./data/embeddings/imagenet_t_50k_embeddings.npz"
elif [ $external_dataset == "imgnet_t_100k" ]; then
    embedding_path="./data/embeddings/imagenet_t_100k_embeddings.npz"
elif [ $external_dataset == "imgnet_t_150k" ]; then
    embedding_path="./data/embeddings/imagenet_t_150k_embeddings.npz"
fi

echo ************************************************************
echo "external dataset: $external_dataset"
echo "embedding_path: $embedding_path"
echo ************************************************************

CUDA_VISIBLE_DEVICES=$gpu_num python -u -m torch.distributed.launch \
    --master_port=$port --nproc_per_node=$nproc_per_node \
    --use_env ./ra_touch/train.py \
    --data_config ./configs/finetune-data-config.yaml \
    --epochs 1 \
    --warmup_epochs 1 \
    --batch_size 1 \
    --accum_iter 4 \
    --blr 10e-4 \
    --weight_decay 0.02 \
    --llama_type llama-2-7b \
    --phase "retrieval" \
    --llama_path $llama_path \
    --output_dir $output_dir  \
    --pretrained_path $pretrained_path \
    --active_modality_names tactile vision \
    --checkpoint_path $encoder_ckpt_path \
    --tactile_model $tactile_model \
    --log_name $log_name \
    --embedding_path $embedding_path \
    --crop_tacvis \
    --top_k $topk \
    --retrieval_method $retrieval_method \
    --retriever_weight $retriever_weight