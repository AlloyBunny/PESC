# PESC

[English](README.md) | 中文

## 环境配置

本项目需要配置两个独立的 conda 环境。环境名称请保持为 `cfbench` 和 `swift`，项目中的脚本会自动激活对应环境。

### 1. 环境：cfbench

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

### 2. 环境：swift

```bash
conda create -n swift python=3.10
conda activate swift
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

## 数据准备

### 1. 下载数据

1. 从以下链接下载数据集：[数据下载](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)
2. 解压到项目根目录

目录结构如下：

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

### 2. 模型准备

训练和推理需要以下基础模型：
- Qwen2.5-7B-Instruct
- Llama3.1-8B-Instruct

请将模型下载至 `models/` 目录。

## 快速开始

### 训练

#### 方式一：使用训练脚本

1. 进入训练脚本目录：
   - `train_script/dpo/` 目录包含与论文设置一致的 DPO 训练脚本
   - `train_script/sft/` 目录包含 SFT 训练脚本

2. 运行训练脚本。例如，使用英文数据对 Llama3.1-8B-Instruct 进行 DPO 训练：

```bash
bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
```

#### 方式二：使用预训练模型

1. `checkpoint/` 目录包含四个已训练的 DPO 模型的 LoRA 权重文件
2. 使用 `train_script/merge_lora.sh` 合并 LoRA 权重，获得完整的 DPO 模型

### 推理

#### 步骤 1：配置环境变量

1. 将 `env.example` 重命名为 `.env`
2. 根据实际需求配置 API 密钥

#### 步骤 2：本地模型推理（可选）

如需使用本地模型进行推理：

1. 下载对应的基础模型
2. 使用 `train_script/merge_lora.sh` 合并 LoRA 权重（详见"训练"部分）
3. 运行对应的部署脚本（`inference_scripts/deploy_*.sh`）启动模型服务
4. 运行推理脚本（`inference_scripts/inference_*.sh`）

**重要提示**：使用本地模型进行推理前，必须先运行对应的 `deploy_*.sh` 脚本启动模型服务。

#### 步骤 3：运行推理脚本

- `inference_scripts/deploy_*.sh`：使用 vLLM 部署模型服务
- `inference_scripts/inference_*.sh`：执行评测脚本

## 其他脚本

以下目录包含消融实验和分析实验的脚本：

- `inference_scripts/ablation1(prt_levels)/`：个性化反思层级消融实验
- `inference_scripts/analyse1(cross_user)/`：跨用户泛化分析
- `inference_scripts/analyse3(topic_analyse)/`：主题分析实验
