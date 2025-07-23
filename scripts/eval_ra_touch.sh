gpu_num=$1
vit_type=$2
retriever_ckpt=$3
ra_touch_ckpt=$4
topk=${5:-5}
external_dataset=${6:-"imgnet_t_150k"}
retrieval_method=${7:-"txt2txt"}
port=${8:-1113}

if [ -f "scripts/openai_key.txt" ]; then
    OPENAI_KEY=$(cat scripts/openai_key.txt | tr -d '[:space:]')
else
    echo "Error: openai_key.txt not found."
    exit 1
fi

llama_path="./llama-2"
output_dir="./output"
dataset_dir=/your/dataset/path

if [ $vit_type == "tiny" ]; then
    pretrained_path="./weights/tvl_llama_vittiny.pth"
    tactile_model="vit_tiny_patch16_224"
    encoder_ckpt_path="./weights/tvl_enc_vittiny.pth"
elif [ $vit_type == "small" ]; then
    pretrained_path="./weights/tvl_llama_vits.pth"
    tactile_model="vit_small_patch16_224"
    encoder_ckpt_path="./weights/tvl_enc_vits.pth"
elif [ $vit_type == "base" ]; then
    pretrained_path="./weights/tvl_llama_vitb.pth"
    tactile_model="vit_base_patch16_224"
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

CUDA_VISIBLE_DEVICES=$device OPENAI_API_KEY=$gpt_key python ./ra_touch/evaluate.py \
    --has_lora --model_path $ra_touch_ckpt --gpt \
    --active_modality_names tactile vision \
    --tactile_model $tactile_model --crop_tacvis \
    --checkpoint_path $encoder_ckpt_path --eval_datasets ssvtp hct \
    --embedding_path $embedding_path \
    --datasets_dir $dataset_dir --llama_dir $llama_dir \
    --output_dir ./results \
    --phase $phase \
    --top_k $topk \
    --retrieval_method $retrieval_method \
    --retriever_weight $retriever_ckpt
    