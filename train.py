import numpy as np
import torch, sys, argparse, time
import torch.optim as optim
import torch.nn as nn

import model as md
import dataset as ds

SEED  = 0
NUM_GPUS = None


def train(model_handler, num_epochs, verbose=True, dev_data=None):
    '''
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation starting
    after 10 epochs. Saves at most 10 checkpoints plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    :param corpus_samplers: list of samplers for individual corpora, None
                            if only evaling on the full corpus.
    '''
    for epoch in range(num_epochs):
        model_handler.train_step()
        if verbose:
            print("training loss: {}".format(model_handler.loss))
            dev_scores = model_handler.eval_and_print(data=dev_data, data_name='DEV')
            model_handler.save_best(scores=dev_scores)

    model_handler.save(num='FINAL')
    model_handler.eval_and_print(data_name='TRAIN')
    if dev_data is not None:
        model_handler.eval_and_print(data=dev_data, data_name='DEV')

batch_size = 64
checkpoint_path = 'checkpoints/'
epochs = 9
result_path = 'results/'
hidden_size = 275
in_dropout = 0.3
lr = .0006
max_tok_len = 200
max_top_len = 20
early_stop = 0
name = 'gte-base-ffnn'
bias=False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False, default='./data/stance/train/train.csv')
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', required=False, default='./data/stance/val/val.csv')
    parser.add_argument('-n', '--name', help='something to add to the saved model name',
                        required=False, default='')
    parser.add_argument('-k', '--score_key', help='Score to use for optimization', required=False,
                        default='f_macro')
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    data = ds.StanceDataGTE(args['trn_data'], max_tok_len=max_tok_len,
                                   max_top_len=max_top_len,
                                   add_special_tokens=False)

    dataloader = ds.DataSampler(data,  batch_size=batch_size)
   
    dev_data = ds.StanceDataGTE(args['dev_data'], max_tok_len=max_tok_len,
                                           max_top_len=max_top_len,
                                           add_special_tokens=False)

    dev_dataloader = ds.DataSampler(dev_data, batch_size=batch_size, shuffle=False)

    input_layer = md.GTELayer()

    setup_fn = ds.setup_helper

    loss_fn = nn.CrossEntropyLoss()
    model = md.FFNN(input_dim=input_layer.dim, in_dropout_prob=in_dropout,
                        hidden_size=hidden_size,
                        add_topic=True, bias=bias)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': ds.prepare_batch_GTE,
                  'name': name,
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn,
                  'fine_tune': False}

    model_handler = md.TorchModelHandler(checkpoint_path=checkpoint_path,
                                         result_path=result_path,
                                         use_score=args['score_key'],
                                         **kwargs)

    start_time = time.time()
    train(model_handler, epochs, dev_data=dev_dataloader)
    print("[{}] total runtime: {:.2f} minutes".format(name, (time.time() - start_time)/60.))