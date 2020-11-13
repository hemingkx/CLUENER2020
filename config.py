import os

data_dir = os.getcwd() + '/NER/dataset/cluener/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
vocab_path = data_dir + 'vocab.npz'

max_vocab_size = 50000000000

batch_size = 32
embedding_size = 128
hidden_size = 384
drop_out = 0.5
lr = 0.001
max_epoch = 10
lr_decay = 0.95
epochs = 1

label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30,
    "<START>": 31,
    "<STOP>": 32
}

id2label = {_id: _label for _label, _id in list(label2id.items())}

if __name__ == '__main__':
    print(data_dir)
