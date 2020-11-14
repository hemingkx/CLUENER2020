def find_entities(xs, ys, id2word, id2label, label_type, res=[]):
    """get entities in one sentence x with label y"""
    entity = []
    # 将softmax值转换为判别值
    # print("ys: ", ys)
    if label_type == "pred":
        ys = ys.argmax(dim=2)
    # tensor -> list
    xs = xs.tolist()
    ys = ys.tolist()
    for x, y in zip(xs, ys):
        for i in range(len(x)):
            # print(x[i])
            # print(y[i])
            if x[i] == 'O' or y[i] == 'O':
                continue
            if id2label[y[i]][0] == 'B':
                # entity: word + label, 比如"酒/B-position"
                entity = [id2word[x[i]] + '/' + id2label[y[i]]]
            # label类型一样，才会append，比如都是position
            elif id2label[y[i]][0] == 'I' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2label[y[i]][1:]:
                entity.append(id2word[x[i]] + '/' + id2label[y[i]])
            elif id2label[y[i]][0] == 'S' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2label[y[i]][1:]:
                entity.append(id2word[x[i]] + '/' + id2label[y[i]])
                entity.append(str(i))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def f1_score(data_loader, id2word, id2label, model, device):
    entity_pred = []
    entity_label = []
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, lens = batch_samples
        sentences = sentences.to(device)
        labels = labels.to(device)
        label_pred = model.forward(sentences)
        entity_pred = find_entities(sentences, label_pred, id2word, id2label, "pred", entity_pred)
        entity_label = find_entities(sentences, labels, id2word, id2label, "label", entity_label)
    entity_right = [i for i in entity_pred if i in entity_label]
    if len(entity_right) != 0:
        acc = float(len(entity_right)) / len(entity_pred)
        recall = float(len(entity_right)) / len(entity_label)
        return (2 * acc * recall) / (acc + recall)
    else:
        return 0
