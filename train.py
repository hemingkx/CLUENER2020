import torch
from torch.utils.data import DataLoader

import config
from data_loader import NERDataset
from calculate import f1_score

import numpy as np

# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)


def train(train_loader, dev_loader, vocab, model, loss_function, optimizer, device):
    """train the model and test model performance"""
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
                    # dev loss calculation
                    dev_loss, f1 = dev(dev_loader, vocab, model, loss_function, device)
                print("epoch: ", epoch, ", index: ", idx, ", train loss: ", loss.item(),
                      ", f1 score: ", f1, ", dev loss: ", dev_loss)
    print("Training Finished!")


def sample_test(test_input, test_label, model, device):
    """test model performance on a specific sample"""
    test_input = test_input.to(device)
    tag_scores = model.forward(test_input)
    tag_scores = tag_scores.argmax(dim=2)
    print(test_label)
    print(np.array(tag_scores.cpu()))


def dev(dev_loader, vocab, model, loss_function, device):
    """test model performance on dev-set"""
    dev_loss = 0
    for _, test_samples in enumerate(dev_loader):
        x_test, y_test, lens_ = test_samples
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        model.zero_grad()
        y_pred = model.forward(x_test)
        y_pred = y_pred.permute(0, 2, 1)
        # 计算梯度
        dev_loss += loss_function(y_pred, y_test)
    dev_loss = float(dev_loss) / len(dev_loader)
    # f1_score calculation
    f1 = f1_score(dev_loader, vocab.id2word, vocab.id2label, model, device)
    return dev_loss, f1


def test(dataset_dir, vocab, model, loss_function, device):
    """test model performance on the final test set"""
    data = np.load(dataset_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    # build dataset
    test_dataset = NERDataset(word_test, label_test, vocab, config.label2id)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=test_dataset.collate_fn)
    test_loss, f1 = dev(test_loader, vocab, model, loss_function, device)
    print("final test loss: ", test_loss, ", f1 score: ", f1)
