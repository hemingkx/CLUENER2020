import numpy as np
# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)


def find_entities(xs, ys, id2word, id2label, label_type, res=[]):
    """get entities in one sentence x with label y"""
    entity = []
    # 将softmax值转换为判别值
    if label_type == "pred":
        ys = ys.argmax(dim=2)
    # tensor -> list
    xs = xs.tolist()
    ys = ys.tolist()
    for x, y in zip(xs, ys):
        for i in range(len(x)):
            # 数字0对应标签"O"，表示不是标签词
            if x[i] == '0' or y[i] == '0':
                continue
            # 一个多字词的开端
            if id2label[y[i]][0] == 'B':
                # entity: word + label, 比如"酒/B-position"
                entity = [id2word[x[i]] + '/' + id2label[y[i]]]
            # label类型一样，才会append，比如都是position， "I"是一个一个多字词的中间或结尾
            elif id2label[y[i]][0] == 'I' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2label[y[i]][1:]:
                entity.append(id2word[x[i]] + '/' + id2label[y[i]])
                if i == len(x) - 1:
                    entity.append(str(i))
                    res.append(entity)
                    entity = []
                elif id2label[y[i+1]][0] != 'I' or entity[-1].split('/')[1][1:] != id2label[y[i]][1:]:
                    entity.append(str(i))
                    res.append(entity)
                    entity = []
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
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, lens = batch_samples
        #print("labels", np.array(labels))
        sentences = sentences.to(device)
        labels = labels.to(device)
        label_pred = model.forward(sentences)
        entity_pred = find_entities(sentences, label_pred, id2word, id2label, "pred", entity_pred)
        entity_label = find_entities(sentences, labels, id2word, id2label, "label", entity_label)
    entity_right = [i for i in entity_pred if i in entity_label]
    print("entity_pred: ", len(entity_pred), "entity_label: ", len(entity_label), "entity_right: ", len(entity_right))
    if len(entity_right) != 0:
        acc = float(len(entity_right)) / len(entity_pred)
        recall = float(len(entity_right)) / len(entity_label)
        return (2 * acc * recall) / (acc + recall)
    else:
        return 0
