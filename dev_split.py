import numpy as np
from sklearn.model_selection import train_test_split

import config
from data_process import Processor
from Vocabulary import Vocabulary


def dev_split(dataset_dir):
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=0.1, random_state=0)
    return x_train, x_dev, y_train, y_dev


if __name__ == "__main__":
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    print(len(word_train))