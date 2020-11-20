"""split the conll dataset for our model and build tags"""
import os
import random
import argparse
from data_process import Processor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='clue', help="Directory containing the dataset")


def load_dataset(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset) as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                # 读取一句话
                line = line.strip('\n')
                if len(line.split()) > 1:
                    word = line.split()[0]
                    tag = line.split()[-1]
                else:
                    # print(line)
                    continue
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                # 每句话的words和tags就分好了，各自对应用列表存储
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    """Write sentences.txt and tags.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print('Saving in {}...'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
        open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print('- done.')


def build_tags(data_dir, tags_file):
    """
    Build tags from dataset
    Get all the tags of dataset
    """
    data_types = ['train', 'val', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    tags = sorted(tags)
    with open(tags_file, 'w') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = 'data/' + args.dataset

    processor = Processor()
    train = processor.get_examples('train')
    test = processor.get_examples('test')

    # Load the dataset into memory
    print('Loading ' + args.dataset.upper() + ' dataset into memory...')
    # split val
    total_train_len = len(train)
    # train:val = 19:1
    split_val_len = int(total_train_len * 0.05)
    order = list(range(total_train_len))
    random.seed(2019)
    random.shuffle(order)

    # Split the dataset into train, val(split with shuffle) and test
    val = [train[idx] for idx in order[:split_val_len]]
    train = [train[idx] for idx in order[split_val_len:]]
    
    save_dataset(train, data_dir + '/train')
    save_dataset(val, data_dir + '/val')
    save_dataset(test, data_dir + '/test')

    # Build tags from dataset
    build_tags(data_dir, data_dir + '/tags.txt')

