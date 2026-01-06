注：本文件夹下的代码实现了三个分析实验，具体如下

## 分析实验1：跨用户泛化分析（代码：`get_global_PRT.py`中实现了“平均用户画像的生成”，对应.env中的PRT_TYPE=global设置）

前面讲到，训练只用了前90个用户的数据，后10个用户不在训练数据中。本实验对后10个用户分析。

1. 不用PRT
2. 用前90个用户的PRT生成的“平均用户画像”作为PRT（把所有L3做一个平均，得到L3_global作为“平均用户画像”）
3. 使用这10个用户自身数据生成的PRT

（等主实验做完后做这个实验）

## 分析实验2：Checklist Filtering 约束大类分析（代码：`checklist_analyse.py`）

对于一个偏好对$(x,\tilde{y}_i,y_i)$，$x$是上下文，$\tilde{y}_i$是w/ PRT（带个性化记忆）的回复，$y_i$是w/o PRT的回复。

Checklist每个类别$C \in \{性格、偏好、规避\}$，我们统计：

- $N_C^{\text{win}}$：该类别下，“判定$\tilde{y}_i$更好” 的约束条数
- $N_C^{\text{lose}}$：判定$y_i$更好” 的条数
- $N_C^{\text{tie}}$：判定$\tilde{y}_i$和$y_i$差不多好的条数
- $N_C = N_C^{\text{win}} + N_C^{\text{lose}} + N_C^{\text{tie}}$

1. **哪个大类的checklist中，$\tilde{y}_i$对比$y_i$胜率更高？**
   指标：$\text{WinRate}(C) = \frac{N_C^{\text{win}}}{N_C}$，胜率越高说明类别$C$越偏向“判定$\tilde{y}_i$更好”

2. **哪个大类的checklist的$\tilde{y}_i$更好时，该偏好对更容易被接受？（即哪个大类的checklist对于偏好对通过filtering的“贡献”最明显）**

   类别倾向得分：$\begin{array}{c} g_C(i) = 
   \begin{cases}
   \frac{w_C(i) - l_C(i)}{n_C(i)}, & n_C(i) > 0 \\
   0, & n_C(i) = 0
   \end{cases} \end{array}$

   条件通过率：$\text{CAR}(C) = P(\text{accepted} \mid g_C>0) - P(\text{accepted} \mid g_C\le0)$

   （如果要进一步提升严谨性，可以用 Logistic回归 替代CAR，缺点是没那么直观）

```python
# 实验结果
{'性格': {'win': 48294, 'lose': 2323, 'tie': 1526}, '偏好': {'win': 71232, 'lose': 3773, 'tie': 8392}, '规避': {'win': 27916, 'lose': 1345, 'tie': 2084}}
{'性格': {'pos': 24975, 'pos_pass': 23139, 'neg': 1814, 'neg_pass': 105}, '偏好': {'pos': 24845, 'pos_pass': 23177, 'neg': 1944, 'neg_pass': 67}, '规避': {'pos': 23366, 'pos_pass': 21938, 'neg': 3423, 'neg_pass': 1306}}
WinRate(性格)=92.62%
WinRate(偏好)=85.41%
WinRate(规避)=89.06%
CAR(性格)=0.8686033552847224
CAR(偏好)=0.8983987347066215
CAR(规避)=0.5573488964704454
```

## 分析实验3：PRT对不同topic的提升幅度（代码：`statistics_of_topic_classes.py`）

对9个topic大类分别统计w/ PRT和w/o PRT的平均Sentient Score，看PRT“对什么类型的对话提升更大”。这只是对已有结果的统计分析，不用额外做实验，做起来很快。
（等主实验做完后做这个实验）