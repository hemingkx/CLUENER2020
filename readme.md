# Chinese NER Project

本项目为CLUENER2020任务baseline的代码实现，模型包括

- BiLSTM-CRF
- BERT-base + X (softmax/CRF/BiLSTM+CRF)
- Roberta + X (softmax/CRF/BiLSTM+CRF)

本项目BERT-base-X部分的代码编写思路参考 [lemonhu](https://github.com/lemonhu/NER-BERT-pytorch) 。

项目说明参考知乎文章：[用BERT做NER？教你用PyTorch轻松入门Roberta！](https://zhuanlan.zhihu.com/p/346828049)

## Dataset

实验数据来自[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)。这是一个中文细粒度命名实体识别数据集，是基于清华大学开源的文本分类数据集THUCNEWS，选出部分数据进行细粒度标注得到的。该数据集的训练集、验证集和测试集的大小分别为10748，1343，1345，平均句子长度37.4字，最长50字。由于测试集不直接提供，考虑到leaderboard上提交次数有限，**本项目使用CLUENER2020的验证集作为模型表现评判的测试集**。

CLUENER2020共有10个不同的类别，包括：组织(organization)、人名(name)、地址(address)、公司(company)、政府(government)、书籍(book)、游戏(game)、电影(movie)、职位(position)和景点(scene)。

原始数据分别位于具体模型的/data/clue/路径下，train.json和test.json文件中，文件中的每一行是一条单独的数据，一条数据包括一个原始句子以及其上的标签，具体形式如下：

```
{
	"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
	"label": {
		"name": {
			"叶老桂": [
				[9, 11],
				[32, 34]
			]
		},
		"company": {
			"浙商银行": [
				[0, 3]
			]
		}
	}
}

```

该数据集的数据在标注时，由于需要保证数据的真实性存在一些质量问题，参见：[数据问题一](https://github.com/CLUEbenchmark/CLUENER2020/issues/10)、[数据问题二](https://github.com/CLUEbenchmark/CLUENER2020/issues/8)，对整体没有太大影响。

## Model

CLUENER2020官方的排行榜：[传送门](https://www.cluebenchmarks.com/ner.html)。

本项目实现了CLUENER2020任务的baseline模型，对应路径分别为：

- BiLSTM-CRF
- BERT-Softmax
- BERT-CRF
- BERT-LSTM-CRF

其中，根据使用的预训练模型的不同，BERT-base-X 模型可转换为 Roberta-X 模型。

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- scikit-learn
- pytorch >= 1.5.1
- 🤗transformers == 2.2.2

To get the environment settled, run:

```
pip install -r requirements.txt
```

## Pretrained Model Required

需要提前下载BERT的预训练模型，包括

- pytorch_model.bin
- vocab.txt

放置在./pretrained_bert_models对应的预训练模型文件夹下，其中

**bert-base-chinese模型：**[下载地址](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 。

注意，以上下载地址仅提供tensorflow版本，需要根据[huggingface suggest](https://huggingface.co/transformers/converting_tensorflow_models.html)将其转换为pytorch版本。

**chinese_roberta_wwm_large模型：**[下载地址](https://github.com/ymcui/Chinese-BERT-wwm#%E4%BD%BF%E7%94%A8%E5%BB%BA%E8%AE%AE) 。

如果觉得麻烦，pytorch版本的上述模型可以通过下方**网盘链接**直接获取😊：

链接: https://pan.baidu.com/s/1rhleLywF_EuoxB2nmA212w  密码: isc5

## Results

各个模型在数据集上的结果（f1 score）如下表所示：（Roberta均指RoBERTa-wwm-ext-large模型）

| 模型         | BiLSTM+CRF | Roberta+Softmax | Roberta+CRF | Roberta+BiLSTM+CRF |
| ------------ | ---------- | --------------- | ----------- | ------------------ |
| address      | 47.37      | 57.50           | **64.11**   | 63.15              |
| book         | 65.71      | 75.32           | 80.94       | **81.45**          |
| company      | 71.06      | 76.71           | 80.10       | **80.62**          |
| game         | 76.28      | 82.90           | 83.74       | **85.57**          |
| government   | 71.29      | 79.02           | **83.14**   | 81.31              |
| movie        | 67.53      | 83.23           | 83.11       | **85.61**          |
| name         | 71.49      | 88.12           | 87.44       | **88.22**          |
| organization | 73.29      | 74.30           | 80.32       | **80.53**          |
| position     | 72.33      | 77.39           | **78.95**   | 78.82              |
| scene        | 51.16      | 62.56           | 71.36       | **72.86**          |
| **overall**  | 67.47      | 75.90           | 79.34       | **79.64**          |

## Parameter Setting

### 1.model parameters

在./experiments/clue/config.json中设置了Bert/Roberta模型的基本参数，而在./pretrained_bert_models下的两个预训练文件夹中，config.json除了设置Bert/Roberta的基本参数外，还设置了'X'模型（如LSTM）参数，可根据需要进行更改。

### 2.other parameters

环境路径以及其他超参数在./config.py中进行设置。

## Usage

打开指定模型对应的目录，命令行输入：

```
python run.py
```

模型运行结束后，最优模型和训练log保存在./experiments/clue/路径下。在测试集中的bad case保存在./case/bad_case.txt中。

## Attention

目前，当前模型的train.log已保存在./experiments/clue/路径下，如要重新运行模型，请先将train.log移出当前路径，以免覆盖。

