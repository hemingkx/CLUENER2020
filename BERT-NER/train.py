"""Train and evaluate the model"""
import os
import torch
import utils
import random
import logging
import argparse
import torch.nn as nn
from tqdm import trange
from evaluate import evaluate
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='clue', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., "
                         "'experiments/clue/'")


def train_epoch(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        # fetch the next training batch
        batch_data, batch_token_starts, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)  # get padding mask

        # compute model output and loss
        loss = \
        model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir, restore_dir=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        model = BertForSequenceTagging.from_pretrained(tagger_model_dir)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

        # Train for one epoch on training set
        train_epoch(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        # train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        # params.eval_steps = params.train_steps
        # train_metrics = evaluate(model, train_data_iterator, params, mark='Train') # callback train f1
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, val_data_iterator, params, mark='Val')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    args = parser.parse_args()
    tagger_model_dir = 'experiments/' + args.dataset

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'train.log'))
    logging.info("device: {}".format(params.device))

    # Create the input data pipeline

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset

    if args.dataset in ["clue"]:
        # bert_class = 'bert-base-chinese' # auto
        bert_class = 'pretrained_bert_models/bert-base-chinese/'  # manual

    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    logging.info("Loading the datasets...")

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("Loading BERT model...")

    # Prepare model
    model = BertForSequenceTagging.from_pretrained(bert_class, num_labels=len(params.tag2idx))
    model.to(params.device)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': params.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:  # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate, correct_bias=False)
    train_steps_per_epoch = params.train_size // params.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=params.epoch_num * train_steps_per_epoch)

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, tagger_model_dir, args.restore_dir)
