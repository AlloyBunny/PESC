# PESC

English | [中文](README_ZH.md)

## Environment Setup

This project requires two separate conda environments. Please keep the environment names as `cfbench` and `swift`, as the project scripts automatically activate the corresponding environments.

### 1. Environment: cfbench

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

### 2. Environment: swift

```bash
conda create -n swift python=3.10
conda activate swift
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

## Data Preparation

### 1. Download Data

1. Download the dataset from the following link: [Data Download](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)
2. Extract the archive to the project root directory

The directory structure should be as follows:

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

### 2. Model Preparation

The following base models are required for training and inference:
- Qwen2.5-7B-Instruct
- Llama3.1-8B-Instruct

Download the models and place them in the `models/` directory.

## Quick Start

### Training

#### Option 1: Using Training Scripts

1. Navigate to the training script directory:
   - `train_script/dpo/` contains DPO training scripts with the same settings as in the paper
   - `train_script/sft/` contains SFT training scripts

2. Run the training script. For example, to perform DPO training on Llama3.1-8B-Instruct using English data:

```bash
bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
```

#### Option 2: Using Pre-trained Models

1. The `checkpoint/` directory contains LoRA weight files for four trained DPO models
2. Use `train_script/merge_lora.sh` to merge the LoRA weights and obtain the complete DPO models

### Inference

#### Step 1: Configure Environment Variables

1. Rename `env.example` to `.env`
2. Configure API keys according to your needs

#### Step 2: Local Model Inference (Optional)

If you want to use local models for inference:

1. Download the corresponding base models
2. Use `train_script/merge_lora.sh` to merge LoRA weights (see the "Training" section for details)
3. Run the corresponding deployment script (`inference_scripts/deploy_*.sh`) to start the model service
4. Run the inference script (`inference_scripts/inference_*.sh`)

**Important**: Before using local models for inference, you must first run the corresponding `deploy_*.sh` script to start the model service.

#### Step 3: Run Inference Scripts

- `inference_scripts/deploy_*.sh`: Deploy model services using vLLM
- `inference_scripts/inference_*.sh`: Execute evaluation scripts

## Additional Scripts

The following directories contain scripts for ablation studies and analysis experiments:

- `inference_scripts/ablation1(prt_levels)/`: Ablation study on personalized reflection levels
- `inference_scripts/analyse1(cross_user)/`: Cross-user generalization analysis
- `inference_scripts/analyse3(topic_analyse)/`: Topic analysis experiments
