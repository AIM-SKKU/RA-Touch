# RA-Touch

> **[RA-Touch: Retrieval-Augmented Touch Understanding with Enriched Visual Data](https://arxiv.org/abs/2505.14270)**<br>
> [Yoorhim Cho](https://ofzlo.github.io/)<sup>\*</sup>, [Hongyeob Kim](https://redleaf-kim.github.io/)<sup>\*</sup>, [Semin Kim](https://sites.google.com/g.skku.edu/semin-kim), [Youjia Zhang](https://youjia-zhang.github.io/), [Yunseok Choi](https://choiyunseok.github.io/), [Sungeun Hong](https://www.csehong.com/)<sup>†</sup> <br>
> \* Denotes equal contribution <br>
> Sungkyunkwan University <br>

<a href="https://aim-skku.github.io/RA-Touch/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
<a href="https://arxiv.org/abs/2505.14270"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:RA-Touch&color=red&logo=arxiv"></a> &ensp;

<p align="center">
  <img src="/images/teaser.jpg" width="550px">
</p>

## Abstract
### TL;DR
> We introduce RA-Touch, a retrieval-augmented framework that improves visuo-tactile perception by leveraging visual data enriched with tactile semantics. We carefully recaption a large-scale visual dataset, ImageNet-T, with tactile-focused descriptions, enabling the model to access tactile semantics typically absent from conventional visual datasets.
<details><summary>FULL abstract</summary>
Visuo-tactile perception aims to understand an object’s tactile properties, such as texture, softness, and rigidity. However, the field
remains underexplored because collecting tactile data is costly and labor-intensive. We observe that visually distinct objects can exhibit similar surface textures or material properties. For example, a leather sofa and a leather jacket have different appearances but share similar tactile properties. This implies that tactile understanding can be guided by material cues in visual data, even without direct tactile supervision. In this paper, we introduce RA-Touch, a retrieval-augmented framework that improves visuo-tactile perception by leveraging visual data enriched with tactile semantics. We carefully recaption a large-scale visual dataset with tactile-focused descriptions, enabling the model to access tactile semantics typically absent from conventional visual datasets. A key challenge remains in effectively utilizing these tactile-aware external descriptions. RATouch addresses this by retrieving visual-textual representations aligned with tactile inputs and integrating them to focus on relevant textural and material properties. By outperforming prior methods
on the TVL benchmark, our method demonstrates the potential of retrieval-based visual reuse for tactile understanding.
</details>

## Instructions
### Setup
```bash
conda create -n ra-touch python=3.10 -y
conda activate ra-touch
git clone https://github.com/AIM-SKKU/RA-Touch.git
cd RA-Touch
```
#### with pip:
```bash
pip install -r requirements.txt
pip install -e .
```

#### with uv:
```bash
uv wync
source .venv/bin/acivate
```

**Download Pre-trained Weights:**
Please download the required pre-trained weights from the [TVL repository](https://github.com/Max-Fu/tvl):
- TVL Encoder weights (e.g., `tvl_enc_vitb.pth`, `tvl_enc_vits.pth`, `tvl_enc_vittiny.pth`)
- TVL-LLaMA weights (e.g., `tvl_llama_vitb.pth`, `tvl_llama_vits.pth`, `tvl_llama_vittiny.pth`)

Place these weights in the `./weights/` directory.

We have used `torch 2.5.1` for training and evaluation on A6000 GPUs.

### Tactile-Guided Retriever
#### Training
To train the tactile-guided retriever:
```bash
bash scripts/train_retriever.sh <gpu_ids> <vit_type> <epochs> <batch_size> <port>
# bash scripts/train_retriever.sh 0,1 base 60 256 23500
```

Arguments:
- `gpu_ids`: Comma-separated GPU IDs (e.g., "0,1,2,3")
- `vit_type`: Vision Transformer type ("tiny", "small", "base") - default: "base"
- `epochs`: Number of training epochs - default: 60
- `batch_size`: Training batch size - default: 256
- `port`: Distributed training port - default: 23500

**Requirements:**
- Pre-trained tactile encoder weights (e.g., `./weights/tvl_enc_vitb.pth`)
- Training data configuration in `configs/finetune-data-config.yaml`

### Texture-Aware Integrator
#### Training
To train the full RA-Touch model with texture-aware integrator:
```bash
bash scripts/train_ra_touch.sh <gpu_ids> <vit_type> <retriever_weight> <topk> <external_dataset> <retrieval_method> <port>
# bash scripts/train_ra_touch.sh 0,1,2,3 base ./output/retriever_checkpoint.pth 5 imgnet_t_150k txt2txt 1113
```

Arguments:
- `gpu_ids`: Comma-separated GPU IDs
- `vit_type`: Vision Transformer type ("tiny", "small", "base")
- `retriever_weight`: Path to trained retriever checkpoint
- `topk`: Number of top-k retrieved samples - default: 5
- `external_dataset`: External dataset for retrieval ("imgnet_t_10k", "imgnet_t_50k", "imgnet_t_100k", "imgnet_t_150k")
- `retrieval_method`: Retrieval method - default: "txt2txt"
- `port`: Distributed training port - default: 1113

**Requirements:**
- LLaMA-2 model in `./llama-2/` directory
- Pre-trained TVL-LLaMA weights (e.g., `./weights/tvl_llama_vitb.pth`)
- ImageNet-T embeddings (e.g., `./data/embeddings/imagenet_t_150k_embeddings.npz`)

#### Evaluation on TVL-Benchmark
To evaluate the trained RA-Touch model:
```bash
bash scripts/eval_ra_touch.sh <gpu_ids> <vit_type> <retriever_ckpt> <ra_touch_ckpt> <topk> <external_dataset> <retrieval_method> <port>
# bash scripts/eval_ra_touch.sh 0 base ./output/retriever_checkpoint.pth ./output/ra_touch_checkpoint.pth 5 imgnet_t_150k txt2txt 1113
```

**Requirements:**
- OpenAI API key in `scripts/openai_key.txt` for GPT evaluation
- TVL dataset path configured in the script
- Trained retriever and RA-Touch model checkpoints

We follow the same paired t-test as [TVL](https://github.com/Max-Fu/tvl). Please replace the OpenAI API key in `scripts/openai_key.txt` after obtaining your [OPENAI_API_Key](https://platform.openai.com/api-keys)

## Dataset
TBU

## Models
TBU

## To-Do List
- [ ] Training Code
- [ ] Release ImgeNet-T Dataset

## Acknowledgement
This repository is built using the [TVL](https://github.com/Max-Fu/tvl) repository.

## 📖BibTeX
```bibtex
@article{cho2025ra,
  title={RA-Touch: Retrieval-Augmented Touch Understanding with Enriched Visual Data},
  author={Cho, Yoorhim and Kim, Hongyeob and Kim, Semin and Zhang, Youjia and Choi, Yunseok and Hong, Sungeun},
  journal={arXiv preprint arXiv:2505.14270},
  year={2025}
}
```
