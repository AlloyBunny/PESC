# PESC

English | [中文](README_ZH.md)

## Environment Setup

This project requires two separate conda environments.

### Environment 1: cfbench

```bash
conda create -n cfbench python=3.11
conda activate cfbench
pip install argparse==1.4.0
pip install arrow==1.3.0
pip install openai==1.99.9
pip install transformers==4.56.1
pip install requests==2.32.3
pip install tqdm==4.66.4
pip install zhipuai==2.1.4.20230809.1
pip install vllm
pip install chromadb
```

### Environment 2: swift

```bash
conda create -n swift python=3.10
conda activate swift
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

**Note**: Please keep the environment names as `cfbench` and `swift`, as the project scripts automatically activate the corresponding environments.

## Data Preparation

### Download Data

Please download the dataset from the following link: [Data Download](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)

After downloading, extract the archive to the project root directory. The directory structure should be as follows:

```
PESC_Data/
├── checkpoint/
├── dataset/
├── individual_memory/
│   ├── memory_chroma_db_en/
│   ├── memory_chroma_db_zh/
│   ├── en/
│   └── zh/
└── profile/
```

### Model Preparation

The following base models are required for training and inference:
- Qwen2.5-7B-Instruct
- Llama3.1-8B-Instruct

Please download the models and place them in the `models/` directory.

## Quick Start

### Training

#### Using Training Scripts

The `train_script/dpo/` directory contains DPO training scripts with the same settings as in the paper. The `train_script/sft/` directory contains SFT training scripts.

For example, to perform DPO training on Llama3.1-8B-Instruct using English data:

```bash
bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
```

#### Using Pre-trained Models

The `checkpoint/` directory contains LoRA weight files for four trained DPO models. You can directly use `train_script/merge_lora.sh` to merge the LoRA weights and obtain the complete DPO models.

### Inference

#### Configure Environment Variables

1. Rename `env.example` to `.env`
2. Configure API keys and model paths according to your needs

#### Local Model Inference

1. Download the corresponding base models
2. Use `train_script/merge_lora.sh` to merge LoRA weights (see the "Training" section for details)
3. Run the corresponding deployment script (`inference_scripts/deploy_*.sh`)
4. Run the inference script (`inference_scripts/inference_*.sh`)

**Important**: Before using local models for inference, you must first run the corresponding `deploy_*.sh` script to start the model service.

#### Inference Scripts

- `inference_scripts/deploy_*.sh`: Deploy model services using vLLM
- `inference_scripts/inference_*.sh`: Execute evaluation scripts

## Additional Scripts

The following directories contain scripts for ablation studies and analysis experiments:

- `inference_scripts/ablation1(prt_levels)/`: Ablation study on personalized reflection levels
- `inference_scripts/analyse1(cross_user)/`: Cross-user generalization analysis
- `inference_scripts/analyse3(topic_analyse)/`: Topic analysis experiments
