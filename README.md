# ChID_baseline
计算语言学22-23学年秋季学期课程大作业baseline实现

本次作业采用ChID中文成语完型填空数据集，要求模型能够从一系列成语候选中选择最正确的成语填入语篇的特定位置。
```
{
	"groundTruth": ["一目了然", "先入为主"], 
	"candidates": [["明明白白", "添油加醋", "一目了然", "残兵败将", "杂乱无章", "心中有数", "打抱不平"], ["矫揉造作", "死不瞑目", "先入为主", "以偏概全", "期期艾艾", "似是而非", "追根究底"]], 
	"content": "【分析】父母对孩子的期望这一点可以从第一段中找到“即使是学校也只是我们送孩子去接受实用教育的地方，而不是让他们为了知识而去追求知识的地方。”至此，答案选项[C]#idiom#。而选项[B]显然错误。选项[A]这个干扰项是出题人故意拿出一个本身没有问题，但是不适合本处的说法来干扰考生。考生一定要警惕#idiom#的思维模式，在做阅读理解的时候，不能按照自己的直觉和知识瞎猜，一定要以原文为根据。选项[D]显然也是不符合家长的期望的。", 
	"realCount": 2
}
```
如上所示，在`content`中有两处`#idiom#`标志。以第一个标志处为例，模型要从候选成语`["明明白白", "添油加醋", "一目了然", "残兵败将", "杂乱无章", "心中有数", "打抱不平"]`选择最合适的成语`一目了然`填入此处。

## 数据集下载
本文数据集ChID由 **[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075)** 提出。训练集包含句子50w条，验证集和测试集各有2w条句子。

[下载链接（北大网盘）](https://disk.pku.edu.cn:443/link/3510A73BA4793A830B0179DF795330C8)

考虑到同学们的训练资源各不相同，我们鼓励大家在1w，5w，10w三种训练集规格中**选择一种**进行实验；相应的训练集均以提前切分好，放在网盘以供大家下载。

## 模型介绍
baseline模型基于预训练模型中常用的掩码语言模型（Masked Language Modeling）实现，采用中文roberta作为backbone。

我们将`#idiom#`替换为`[MASK][MASK][MASK][MASK]`，通过LM head输出每个`[MASK]`处的token在候选字上的概率分布，并以此选择概率最高的成语。

例如对于上方的例子，第一个`[MASK]`处的token将会通过LM head得到所有候选成语第一个字上（明，添，一，残，杂，心，打）的概率分布，对于其他`[MASK]`处的token，也可以用同样方法得到模型在对应7个候选字上的概率分布。

`#idiom#`处填成语一目了然的概率就是每个`[MASK]`处分别填写对应字（一、目、了、然）概率的乘积；候选中概率最高的成语就是最终的预测结果。

具体实现代码参见`model.py`。


## 实验结果
我们提供了我们的baseline模型在不同训练集规模下的准确率（Accuracy）。0代表zero-shot实验，直接使用未经任何fine-tuning的预训练模型进行预测；同学们使用对应规模训练集时可以和对应的baseline结果进行比较。

| #train data |  dev  |  test |
|-------------|:-----:|:-----:|
| 0           | 51.54 | 51.87 |
| 1w          | 64.93 | 64.83 |
| 5w          | 71.94 | 71.92 |
| 10w         | 74.49 | 74.42 |
| full (50w)  | 80.72 | 81.11 |

我们也比较了我们的baseline模型和原论文中基于LSTM的Attentive Reader方法以及人工评测（Human）的表现。baseline相较于human的水平仍有待提升。

| Model                   |  dev  |  test |
|-------------------------|:-----:|:-----:|
| Ours (Roberta)          | 80.72 | 81.11 |
| Attentive Reader (LSTM) | 72.7  | 72.4  |
| Human                   | -     | 87.1  |

## 作业要求
本次作业要求单人或组队完成（最多3人），请每个小组根据任务的特点设计实验。

最终要求每组同学在学期末进行课堂展示，并且提交自己的实验报告和项目代码到课程公邮jsyyyxpku2022@163.com。

大作业的评分将基于课堂展示、实验报告以及模型的表现进行综合评测。
### 实验报告要求
实验报告中应该包含以下内容：
1. 实验目的（阐述任务）
2. 实验原理（描述模型）
3. 实验内容（描述实验步骤，重视可复现性）
4. 实验结果与分析 （描述实验结果并对结果或case进行分析）
5. 实验过程总结和感想（每个组员都要写）
6. 实验分工（写明每个组员的工作量）
### 其他要求与建议
1. 项目作弊会被记0分，包括但不限于抄袭代码、实验造假等。允许使用开源代码，但请在报告中标注哪部分使用了开源代码，并标明来源
2. 禁止以任何形式使用测试集的标签信息
3. 实验结果不是唯一的评判标准，针对任务特点的改进，对失败尝试的深入分析都是加分项
4. 鼓励大家在项目上尝试一些创新的思路，包括但不限于模型的改动，任务形式的转化，从数据本身出发的思考等；在训练数据有限的情况下，如何引入外部知识也是一个值得尝试的方向。
## 参考文献
ChID: A Large-scale Chinese IDiom Dataset for Cloze Test. Chujie Zheng, Minlie Huang, Aixin Sun. ACL (1) 2019: 778-787
It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners. Timo Schick, Hinrich Schütze. NAACL-HLT 2021: 2339-2352

## 有用的网站
https://github.com/chujiezheng/ChID-Dataset
https://github.com/pwxcoo/chinese-xinhua
https://github.com/by-syk/chinese-idiom-db
https://huggingface.co/docs/transformers/index
