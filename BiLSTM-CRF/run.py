import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import utils
import config
import logging
import numpy as np
from model import BiLSTM_CRF
from data_process import Processor
from Vocabulary import Vocabulary
from data_loader import NERDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from train import train, test, sample_test
from sklearn.model_selection import train_test_split

input_array = [[1642, 1291, 40, 2255, 970, 46, 124, 1604, 1915, 547, 0, 173,
                303, 124, 1029, 52, 20, 2839, 2, 2255, 2078, 1553, 225, 540,
                96, 469, 1704, 0, 174, 3, 8, 728, 903, 403, 538, 668,
                179, 27, 78, 292, 7, 134, 2078, 1029, 0, 0, 0, 0,
                0],
               [28, 6, 926, 72, 209, 330, 308, 167, 87, 1345, 1, 528,
                412, 0, 584, 1, 6, 28, 326, 1, 361, 342, 3256, 17,
                19, 1549, 3257, 131, 2, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0],
               [6, 3, 58, 1930, 37, 407, 1068, 40, 1299, 1443, 103, 1235,
                1040, 139, 879, 11, 124, 200, 135, 97, 1138, 1016, 402, 696,
                337, 215, 402, 288, 10, 5, 5, 17, 0, 248, 597, 110,
                84, 1, 135, 97, 1138, 1016, 402, 696, 402, 200, 109, 164,
                0],
               [174, 6, 110, 84, 3, 477, 332, 133, 66, 11, 557, 107,
                181, 350, 0, 70, 196, 166, 50, 120, 26, 89, 66, 19,
                564, 0, 36, 26, 48, 243, 1308, 0, 139, 212, 621, 300,
                0, 444, 720, 4, 177, 165, 164, 2, 0, 0, 0, 0,
                0]]

label_array = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 14, 14, 14, 14, 14,
                14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 4, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 11, 0, 1, 11, 11, 11, 11, 11, 0, 0, 0, 0, 8, 18, 18,
                18, 18, 18, 18, 18, 18, 0, 0, 9, 19, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 8, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 18, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

test_input = torch.tensor(input_array, dtype=torch.long)
test_label = torch.tensor(label_array, dtype=torch.long)


def dev_split(dataset_dir):
    """split one dev set without k-fold"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def k_fold_run():
    """train with k-fold"""
    # set the logger
    utils.set_logger(config.log_dir)
    # 设置gpu为命令行参数指定的id
    if config.gpu != '':
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    logging.info("device: {}".format(device))
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    # 分离出验证集
    data = np.load(config.train_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    kf = KFold(n_splits=config.n_split)
    kf_data = kf.split(words, labels)
    kf_index = 0
    total_test_loss = 0
    total_f1 = 0
    for train_index, dev_index in kf_data:
        kf_index += 1
        word_train = words[train_index]
        label_train = labels[train_index]
        word_dev = words[dev_index]
        label_dev = labels[dev_index]
        test_loss, f1 = run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index)
        total_test_loss += test_loss
        total_f1 += f1
    average_test_loss = float(total_test_loss) / config.n_split
    average_f1 = float(total_f1) / config.n_split
    logging.info("Average test loss: {} , average f1 score: {}".format(average_test_loss, average_f1))


def simple_run():
    """train without k-fold"""
    # set the logger
    utils.set_logger(config.log_dir)
    # 设置gpu为命令行参数指定的id
    if config.gpu != '':
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    logging.info("device: {}".format(device))
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    # simple run without k-fold
    run(word_train, label_train, word_dev, label_dev, vocab, device)


def run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index=0):
    # build dataset
    train_dataset = NERDataset(word_train, label_train, vocab, config.label2id)
    dev_dataset = NERDataset(word_dev, label_dev, vocab, config.label2id)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    # model
    model = BiLSTM_CRF(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       drop_out=config.drop_out,
                       vocab_size=vocab.vocab_size(),
                       target_size=vocab.label_size())
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    # how to initialize these parameters elegantly
    for p in model.crf.parameters():
        _ = torch.nn.init.uniform_(p, -1, 1)
    # train and test
    train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index)
    with torch.no_grad():
        # test on the final test set
        test_loss, f1 = test(config.test_dir, vocab, device, kf_index)
        # sample_test(test_input, test_label, model, device)
    return test_loss, f1


if __name__ == '__main__':
    simple_run()
