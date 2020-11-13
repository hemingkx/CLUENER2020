from data_process import Processor
from Vocabulary import Vocabulary
import config
import numpy as np
from data_loader import NERDataset

if __name__ == '__main__':
    processor = Processor(config)
    processor.data_process()
    # processor.get_examples('sample')
    vocab = Vocabulary(config)
    vocab.get_vocab()
    dataset = NERDataset(config.train_dir, vocab, config.label2id)