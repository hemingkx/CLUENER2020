import torch
from torch.utils.data import DataLoader

import config
import logging
from tqdm import tqdm
from data_loader import NERDataset
from metric import f1_score, bad_case

import numpy as np

# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)


def epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index=0):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        x, y, mask, lens = batch_samples
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        model.zero_grad()
        tag_scores, loss = model.forward_with_crf(x, mask, y)
        train_loss += loss.item()
        # 梯度反传
        loss.backward()
        # 优化更新
        optimizer.step()
        optimizer.zero_grad()
    # scheduler
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    if kf_index == 0:
        logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))
    else:
        logging.info("Kf epoch: {}, epoch: {}, train loss: {}".format(kf_index, epoch, train_loss))


def train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index=0):
    """train the model and test model performance"""
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index)
        with torch.no_grad():
            # dev loss calculation
            metric = dev(dev_loader, vocab, model, device)
            val_f1 = metric['f1']
            dev_loss = metric['loss']
            if kf_index == 0:
                logging.info("epoch: {}, f1 score: {}, "
                             "dev loss: {}".format(epoch, val_f1, dev_loss))
            else:
                logging.info("Kf epoch: {}, epoch: {}, f1 score: {}, "
                             "dev loss: {}".format(kf_index, epoch, val_f1, dev_loss))
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                torch.save(model, config.model_dir)
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_val_f1))
                break
    logging.info("Training Finished!")


def sample_test(test_input, test_label, model, device):
    """test model performance on a specific sample"""
    test_input = test_input.to(device)
    tag_scores = model.forward(test_input)
    labels_pred = model.crf.decode(tag_scores)
    logging.info("test_label: ".format(test_label))
    logging.info("labels_pred: ".format(labels_pred))


def dev(data_loader, vocab, model, device, mode='dev'):
    """test model performance on dev-set"""
    model.eval()
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, masks, lens = batch_samples
        sent_data.extend([[vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                          for (mask, indices) in zip(masks, sentences)])
        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=masks)
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        true_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in targets])
        pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])
        # 计算梯度
        _, dev_loss = model.forward_with_crf(sentences, masks, labels)
        dev_losses += dev_loss
    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(data_loader)
    return metrics


def test(dataset_dir, vocab, device, kf_index=0):
    """test model performance on the final test set"""
    data = np.load(dataset_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    # build dataset
    test_dataset = NERDataset(word_test, label_test, vocab, config.label2id)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=test_dataset.collate_fn)
    # Prepare model
    if config.model_dir is not None:
        # model
        model = torch.load(config.model_dir)
        model.to(device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    metric = dev(test_loader, vocab, model, device, mode='test')
    f1 = metric['f1']
    test_loss = metric['loss']
    if kf_index == 0:
        logging.info("final test loss: {}, f1 score: {}".format(test_loss, f1))
        val_f1_labels = metric['f1_labels']
        for label in config.labels:
            logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
    else:
        logging.info("Kf epoch: {}, final test loss: {}, f1 score: {}".format(kf_index, test_loss, f1))
    return test_loss, f1
