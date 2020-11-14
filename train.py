import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import config
from data_process import Processor
from Vocabulary import Vocabulary
from data_loader import NERDataset
from model import BiLSTM_CRF
from calculate import f1_score
# from sklearn.cross_validation import train_test_split

import numpy as np
# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)

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

if __name__ == "__main__":
    # 设置gpu为命令行参数指定的id
    if config.gpu != '':
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    # build data_loader
    train_dataset = NERDataset(config.train_dir, vocab, config.label2id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    test_dataset = NERDataset(config.test_dir, vocab, config.label2id)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=test_dataset.collate_fn)
    # model
    model = BiLSTM_CRF(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       drop_out=config.drop_out,
                       vocab_size=vocab.vocab_size(),
                       tagset_size=vocab.label_size())
    model.to(device)
    # loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    with torch.no_grad():
        test_input = test_input.to(device)
        tag_scores = model.forward(test_input)
        tag_scores = tag_scores.argmax(dim=2)
        print(test_label)
        print(np.array(tag_scores.cpu()))
    # start training
    for epoch in range(config.epochs):
        # step number in one epoch: 336
        for idx, batch_samples in enumerate(train_loader):
            x, y, lens = batch_samples
            # print("y: ", np.array(y))
            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            y_pred = model.forward(x)
            y_pred = y_pred.permute(0, 2, 1)
            # 计算梯度
            loss = loss_function(y_pred, y)
            # 梯度反传
            loss.backward()
            # 优化更新
            optimizer.step()
            optimizer.zero_grad()
            if idx % 100 == 0:
                with torch.no_grad():
                    for _, test_samples in enumerate(test_loader):
                        x_test, y_test, lens_ = test_samples
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        model.zero_grad()
                        y_pred = model.forward(x_test)
                        y_pred = y_pred.permute(0, 2, 1)
                        # 计算梯度
                        test_loss = loss_function(y_pred, y_test)
                    # f1_score calculation
                    f1 = f1_score(test_loader, vocab.id2word, vocab.id2label, model, device)
                print("epoch: ", epoch, ", index: ", idx, ", train loss: ", loss.item(),
                      ", f1 score: ", f1, ", test loss: ", test_loss.item())
    print("Training Finished!")
    with torch.no_grad():
        test_input = test_input.to(device)
        tag_scores = model.forward(test_input)
        tag_scores = tag_scores.argmax(dim=2)
        print(test_label)
        print(np.array(tag_scores.cpu()))

