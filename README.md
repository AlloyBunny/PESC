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

（注：另一个名为swift的环境是用来训练的，跑评测不需要，所以这里没写）

# 运行

## 主实验

`inference_scripts/`路径下的.sh文件是主实验脚本，deploy开头的脚本用于vllm部署模型，inference开头的是评测脚本。

**==运行本地模型的inference之前必须先运行对应的deploy脚本！==**

## 消融实验和分析实验

`inference_scripts/ablation1(prt_levels)/`、`inference_scripts/analyse1(cross_user)/`、`inference_scripts/analyse3(topic_analyse)/`下的是消融实验、分析实验的脚本。以"inference"开头的脚本和主实验一样，如果跑的本地模型，需要先运行对应的deploy。analyse3的脚本需要在主实验完成之后再运行。