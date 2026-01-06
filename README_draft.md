# 配环境

## 环境1：cfbench

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

## 环境2：swift

```bash
conda create -n swift python=3.10
conda activate swift
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install 'ms-swift'
pip install chromadb
```

说明：两个环境都要配置，并且建议不要改名字，因为我写的一键脚本中带了自动激活对应环境的功能（比如conda activate cfbench）

# 数据下载

在运行之前，要下载[数据](https://drive.google.com/file/d/1rCoou-1xb9SMxSkdQUTmbP0O9TKMczUs/view?usp=sharing)

文件树格式如下所示，下载后解压到项目根目录

```
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



# 快速开始

## 训练

1. 下载Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct模型，放到`models/`目录下。

2. `train_script/dpo`下提供了和论文中设置一样的dpo训练脚本，`train_script/sft`提供了额外的sft训练脚本。
   比如如果要用英文数据对Llama3.1-8B-Instruct进行DPO训练，就要

   ```bash
   bash train_script/dpo/Llama-3.1-8B-Instruct-dpo-en.sh
   ```

3. `checkpint`中包含了四个训练好的DPO模型的LoRA文件，实际上你可以不用自己训练，可以直接使用`train_script/merge_lora.sh`合并lora文件，获得DPO训练后的模型

## 推理

1. 把env.example重命名为.env，并进行配置
2. 如果需要用local model进行推理，你必须先下载对应的base模型，合并LoRA文件以获得DPO模型，详见上一节“训练”。
3. `inference_scripts/`路径下的.sh文件是主实验脚本，deploy开头的脚本用于vllm部署模型，inference开头的是评测脚本。注意，**==运行本地模型的inference之前必须先运行对应的deploy脚本！==**

## 其他脚本说明

`scripts/ablation1(prt_levels)/`、`scripts/analyse1(cross_user)/`、`scripts/analyse3(topic_analyse)/`下的是消融实验、分析实验的脚本。