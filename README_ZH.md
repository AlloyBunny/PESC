# PESC

[English](README.md) | 中文

---

## 环境配置

⚠️ **需要同时配置两个 Conda 环境，并且不建议修改环境名称**，因为部分一键脚本中包含自动激活指定环境的逻辑（例如 `conda activate cfbench`）。

### 环境一：cfbench

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

### 环境二：swift

```bash
conda create -n swift python=3.10
conda activate swift
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

------

## 数据下载

在运行代码之前，请先下载数据集：

👉 [数据下载链接](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)

下载完成后，将压缩包解压到项目根目录，目录结构如下所示：

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

## 快速开始

### 模型训练

1. 下载 **Qwen2.5-7B-Instruct** 和 **Llama3.1-8B-Instruct** 基座模型，并放置在 `models/` 目录下。

2. `train_script/dpo/` 目录下提供了与论文实验设置一致的 DPO 训练脚本，
   `train_script/sft/` 目录下提供了额外的 SFT 训练脚本。

   例如，使用英文数据对 **Llama3.1-8B-Instruct** 进行 DPO 训练：

   ```bash
   bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
   ```

3. `checkpoint/` 目录中已包含 4 个训练完成的 DPO 模型 LoRA 权重。
   如不希望重新训练，可直接运行以下脚本合并 LoRA 权重：

   ```bash
   bash train_script/merge_lora.sh
   ```

   合并后即可获得 DPO 训练完成的模型。

------

### 模型推理

1. 将 `env.example` 重命名为 `.env`，并配置所需模型的名称、API Key 以及对应的服务地址（如 GPT、DeepSeek、Gemini 或本地模型）。

2. 若使用本地模型进行推理，请确保：

   - 已下载对应的基座模型；
   - 已完成 LoRA 权重合并，得到 DPO 模型。

3. `inference_scripts/` 目录下的脚本用于主实验流程：

   - 以 `deploy` 开头的脚本用于使用 **vLLM** 部署模型；
   - 以 `inference` 开头的脚本用于模型评测。

   ⚠️ **在运行本地模型的推理脚本之前，必须先执行对应的 `deploy` 脚本。**

------

## 其他脚本说明

- `scripts/ablation1(prt_levels)/`
  消融实验脚本
- `scripts/analyse1(cross_user)/`
  跨用户分析脚本
- `scripts/analyse3(topic_analyse)/`
  主题分析脚本