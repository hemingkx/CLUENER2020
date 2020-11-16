import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, words, labels, vocab, label2id):
        self.vocab = vocab
        self.dataset = self.preprocess(words, labels)
        self.label2id = label2id

    def preprocess(self, words, labels):
        """convert the data to ids"""
        processed = []
        for (word, label) in zip(words, labels):
            word_id = [self.vocab.word_id(w_) for w_ in word]
            label_id = [self.vocab.label_id(l_) for l_ in label]
            processed.append((word_id, label_id))
        print("--------", "Process Done!", "--------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_long_tensor(tokens_list, batch_size):
        """padding sentence and convert into LongTensor"""
        # get max len of sentences
        token_len = max([len(x) for x in tokens_list])
        tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        # padding
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
        return tokens

    def collate_fn(self, batch):
        """get padding data"""
        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in texts]
        batch_size = len(batch)

        input_ids = self.get_long_tensor(texts, batch_size)
        label_ids = self.get_long_tensor(labels, batch_size)

        return [input_ids, label_ids, lens]
