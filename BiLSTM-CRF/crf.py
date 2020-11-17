import time
import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, device):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        # equal to vocab.label_size
        self.tagset_size = len(tag_to_ix)
        self.device = device
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(self.device)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1).to(self.device)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2).to(self.device)  # +1
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences

        score = torch.zeros(tags.shape[0]).to(self.device)
        tags = torch.cat([torch.full([tags.shape[0], 1],
                                     self.tag_to_ix[START_TAG]).long().to(self.device), tags.to(self.device)], dim=1).to(self.device)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]].to(self.device)
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode_new(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)
        # feats.shape: [11, 6]
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1]).to(self.device)
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0).to(self.device)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        # shape: torch.Size([1, 6])
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        # one number (2 for example)
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            # bptrs_t is like:  [3, 3, 3, 3, 3, 3]
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_new_parallel(self, feats):
        path_scores = []
        best_paths = []
        for index in range(feats.shape[0]):
            path_score, best_path = self._viterbi_decode_new(feats[index])
            best_paths.append(best_path)
            path_scores.append(path_score)
        return path_scores, best_paths

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features_parallel(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new_parallel(lstm_feats)
        return score, tag_seq
