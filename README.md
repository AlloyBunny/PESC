# PESC

English | [中文](README_ZH.md)

---

## Environment Setup

⚠️ **Both environments are required. Do NOT change the environment names**, as the provided scripts rely on automatic environment activation (e.g., `conda activate cfbench`).

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
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

------

## Data Preparation

Before running the code, please download the data from the following link:

👉 [Download data](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)

After downloading, extract the archive to the project root directory. The directory structure should look like:

```text
PESC_Data.tar.gz
├── checkpoint/
├── dataset/
├── individual_memory/
│   ├── memory_chroma_db_en/
│   ├── memory_chroma_db_zh/
│   ├── en/
│   └── zh/
└── profile/
```

------

## Quick Start

### Training

1. Download the base models **Qwen2.5-7B-Instruct** and **Llama3.1-8B-Instruct**, and place them under the `models/` directory.

2. The directory `train_script/dpo/` contains DPO training scripts that match the experimental settings described in the paper.
   Additional SFT training scripts are provided under `train_script/sft/`.

   For example, to perform DPO training on **English data** using **Llama3.1-8B-Instruct**:

   ```bash
   bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
   ```

3. The `checkpoint/` directory already includes LoRA weights for four trained DPO models.
   You may skip training and directly merge the LoRA weights using:

   ```bash
   bash train_script/merge_lora.sh
   ```

   This will produce the final DPO-trained model.

------

### Inference

1. Rename `env.example` to `.env` and configure the API credentials and endpoints for the models you intend to use, including GPT, DeepSeek, Gemini, or local models.

2. To perform inference with local models, ensure that:

   - The corresponding base model has been downloaded.
   - The LoRA weights have been merged to obtain the DPO model.

3. Scripts under `inference_scripts/` are used for the main experiments:

   - Scripts prefixed with `deploy` are used to deploy models with **vLLM**.
   - Scripts prefixed with `inference` are used for evaluation.

   ⚠️ **When using local models, the corresponding `deploy` script must be executed before running inference scripts.**

------

## Additional Scripts

- `scripts/ablation1(prt_levels)/`
  Ablation study scripts
- `scripts/analyse1(cross_user)/`
  Cross-user analysis scripts
- `scripts/analyse3(topic_analyse)/`
  Topic-based analysis scripts