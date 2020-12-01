import os
import logging
import numpy as np


class Vocabulary:
    """
    构建词表
    """
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files
        self.vocab_path = config.vocab_path
        self.max_vocab_size = config.max_vocab_size
        self.word2id = {}
        self.id2word = None
        self.label2id = config.label2id
        self.id2label = config.id2label

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def label_size(self):
        return len(self.label2id)

    # 获取词的id
    def word_id(self, word):
        return self.word2id[word]

    # 获取id对应的词
    def id_word(self, idx):
        return self.id2word[idx]

    # 获取label的id
    def label_id(self, word):
        return self.label2id[word]

    # 获取id对应的词
    def id_label(self, idx):
        return self.id2label[idx]

    def get_vocab(self):
        """
        进一步处理，将word和label转化为id
        word2id: dict,每个字对应的序号
        idx2word: dict,每个序号对应的字
        保存为二进制文件
        """
        # 如果有处理好的，就直接load
        if os.path.exists(self.vocab_path):
            data = np.load(self.vocab_path, allow_pickle=True)
            # '[()]'将array转化为字典
            self.word2id = data["word2id"][()]
            self.id2word = data["id2word"][()]
            logging.info("-------- Vocabulary Loaded! --------")
            return
        # 如果没有处理好的二进制文件，就处理原始的npz文件
        word_freq = {}
        for file in self.files:
            data = np.load(self.data_dir + str(file) + '.npz', allow_pickle=True)
            word_list = data["words"]
            # 常见的单词id最小
            for line in word_list:
                for ch in line:
                    if ch in word_freq:
                        word_freq[ch] += 1
                    else:
                        word_freq[ch] = 1
        index = 0
        sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)
        # 构建word2id字典
        for elem in sorted_word:
            self.word2id[elem[0]] = index
            index += 1
            if index >= self.max_vocab_size:
                break
        # id2word保存
        self.id2word = {_idx: _word for _word, _idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")

