import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import config
from data_process import Processor
from Vocabulary import Vocabulary


def dev_split(dataset_dir):
    """split one dev set without k-fold"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def k_fold_split(dataset_dir):
    """split with k-fold"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    kf = KFold(n_splits=config.n_split)
    kf_data = kf.split(words, labels)
    for train_index, dev_index in kf_data:
        x_train = words[train_index]
        y_train = labels[train_index]
        x_dev = words[dev_index]
        y_dev = labels[dev_index]
        print(len(x_train), len(y_train), len(x_dev), len(y_dev))


if __name__ == "__main__":
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    k_fold_split(config.train_dir)
