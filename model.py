import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_CRF(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, tagset_size, drop_out):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        # self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, tagset_size)

    # self.hidden = self.init_hidden()隐藏层初始化问题？

    # def init_hidden(self):
    # return (torch.zeros(2*2,batch_size,))

    def forward(self, inputs_ids):
        embs = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embs)
        # sequence_output = self.layer_norm(sequence_output)
        features = self.classifier(sequence_output)
        # softmax??
        # batch_size * seq_length *
        tag_scores = F.softmax(features, dim=2)
        # return features
        return tag_scores