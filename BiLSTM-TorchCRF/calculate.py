def find_entities(xs, ys, id2word, id2label, label_type, res=[]):
    """get entities in one sentence x with label y"""
    entity = []
    # tensor -> list
    xs = xs.tolist()
    for x, y in zip(xs, ys):
        for i in range(len(y)):
            # 数字0对应标签"O"，表示不是标签词
            if x[i] == '0' or y[i] == '0':
                continue
            # 一个多字实体的开端
            if id2label[y[i]][0] == 'B':
                # entity: word + label, 比如"酒/B-position"
                entity = [id2word[x[i]] + '/' + id2label[y[i]]]
            # label类型一样，才会append，比如都是position， "I"是一个一个多字实体的中间或结尾
            elif id2label[y[i]][0] == 'I' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2label[y[i]][1:]:
                entity.append(id2word[x[i]] + '/' + id2label[y[i]])
                if i == len(y) - 1:
                    entity.append(str(i))
                    res.append(entity)
                    entity = []
                elif id2label[y[i + 1]][0] != 'I' or entity[-1].split('/')[1][1:] != id2label[y[i]][1:]:
                    entity.append(str(i))
                    res.append(entity)
                    entity = []
            # 一个单字实体的开端
            elif id2label[y[i]][0] == 'S':
                entity = [id2word[x[i]] + '/' + id2label[y[i]]]
                entity.append(str(i))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def f1_score(data_loader, id2word, id2label, model, device):
    entity_pred = []
    entity_label = []
    dev_losses = 0
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, mask, lens = batch_samples
        sentences = sentences.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=mask)
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        # 计算梯度
        _, dev_loss = model.forward_with_crf(sentences, mask, labels)
        dev_losses += dev_loss
        entity_pred = find_entities(sentences, labels_pred, id2word, id2label, "pred",  entity_pred)
        entity_label = find_entities(sentences, targets, id2word, id2label, "pred", entity_label)
    dev_losses = float(dev_losses) / len(data_loader)
    entity_right = [i for i in entity_pred if i in entity_label]
    print("entity_pred: ", len(entity_pred), "entity_label: ", len(entity_label), "entity_right: ", len(entity_right))
    if len(entity_right) != 0:
        acc = float(len(entity_right)) / len(entity_pred)
        recall = float(len(entity_right)) / len(entity_label)
        return (2 * acc * recall) / (acc + recall), dev_losses
    else:
        return 0, dev_losses
