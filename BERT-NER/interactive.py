"Evaluate the model"""
import os
import nltk
import torch
import random
import logging
import argparse
import numpy as np
import utils as utils
from metrics import get_entities
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='msra', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def interAct(model, data_iterator, params, mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    idx2tag = params.idx2tag

    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)
        
    batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)
    batch_output = batch_output.detach().cpu().numpy()
    
    pred_tags = []
    pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
    
    return(get_entities(pred_tags))


def bert_ner_init():
    args = parser.parse_args()
    tagger_model_dir = 'experiments/' + args.dataset

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset
    if args.dataset in ["conll"]:
        bert_class = 'bert-base-cased'
    elif args.dataset in ["msra"]:
        bert_class = 'bert-base-chinese'

    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)

    return model, data_loader, args.dataset, params

def BertNerResponse(model, queryString):    
    model, data_loader, dataset, params = model
    if dataset in ['msra']:
        queryString = [i for i in queryString]
    elif dataset in ['conll']:
        queryString = nltk.word_tokenize(queryString)


    with open('data/' + dataset + '/interactive/sentences.txt', 'w') as f:
        f.write(' '.join(queryString))

    inter_data = data_loader.load_data('interactive')
    inter_data_iterator = data_loader.data_iterator(inter_data, shuffle=False)
    result = interAct(model, inter_data_iterator, params)
    res = []
    for item in result:
        if dataset in ['msra']:
            res.append((''.join(queryString[item[1]:item[2]+1]), item[0]))
        elif dataset in ['conll']:
            res.append((' '.join(queryString[item[1]:item[2]+1]), item[0]))
    return res


def main():
    model = bert_ner_init()
    while True:
        query = input('Input:')
        if query == 'exit':
            break
        print(BertNerResponse(model, query))


if __name__ == '__main__':
    main()


    

