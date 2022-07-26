import sys
import time

import torch  # pytorch == 1.11.0
import torch.nn as nn
import timeit
import torchtext  # torchtext == 0.4.0
import logging
import random
from tqdm import tqdm
from transformers import BertTokenizer
from models import StackedLstm, RNNModelScratch, RNNModelPyTorch, LSTM, GRU
from utils import brewed_dataLoader
from random import choice
import os
import pathlib

from models import RNNModelPyTorch, RNNModelScratch, StackedLstm, CNN
from utils import Accumulator
import math
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# CUDA = 10.2 on the server
def get_device():
    # return torch.device('cpu')
    return torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def create_logger(project_dir: str, file_name: str):
    """
    Generate a logging file and add to logger
    :return: logger
    """

    (pathlib.Path(project_dir) / 'log').mkdir(exist_ok=True)
    print("log file generated: " + file_name)
    file = logging.FileHandler(project_dir + "/log/" + file_name + ".log")
    file.setLevel(logging.INFO)
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file)

    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


def grad_clipping(net, theta):
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def predict(prefix, num_preds, net, vocab, device):
    vocab_itos, vocab_stoi, _ = vocab
    state = None
    outputs = [vocab_stoi[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab_stoi[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

        if outputs[-1] == vocab_stoi['<EOS>']:
            break
    return ' '.join([vocab_itos[i] for i in outputs])


def predict_for_torch_RNN(prefix, num_preds, net, vocab, device, temperature=0.8):
    vocab_itos, vocab_stoi, _ = vocab
    generated_sequence = ""
    # one-to-many
    inp = torch.Tensor([vocab_stoi["<BOS>"]]).long().to(get_device())
    # hidden = model.begin_state(inp.shape[0], get_device())
    for p in range(num_preds):
        output, hidden = net(inp.unsqueeze(0))  # , hidden
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted character to string and use as next input
        predicted_char = vocab_itos[top_i]

        if predicted_char == "<EOS>":
            break
        generated_sequence += " " + predicted_char
        inp = torch.Tensor([top_i]).long().to(get_device())
    return generated_sequence


@torch.no_grad()
def evaluate(net, val_iter, loss, device, vocab):
    metric = Accumulator(2)

    for batch in val_iter:
        seq = batch.content[0]
        batch_size = batch.batch_size
        X = seq[:, :-1]
        Y = seq[:, 1:]

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, _ = net(X)
        l = loss(y_hat.reshape(-1, len(vocab[0])), y.long()).mean()

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1])


def train_epoch(net, train_iter, loss, updater, device, vocab):
    """Train a net within one epoch"""
    start_time = time.time()
    metric = Accumulator(2)  # Sum of training loss, no. of tokens

    for batch in train_iter:
        seq = batch.content[0]
        X = seq[:, :-1]
        Y = seq[:, 1:]

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, _ = net(X)
        l = loss(y_hat, y.long()).mean()

        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

        # this way of computing the perplexity works only because the cross-entropy loss is used!
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), time.time() - start_time


def train(net, train_iter, val_iter, vocab, lr, num_epochs, device, logger=None, experiment_name='experiment'):
    """Train a model"""
    loss = nn.CrossEntropyLoss()
    train_results = []
    val_results = []

    # Initialize
    optimizer = torch.optim.Adam(net.parameters(), lr)

    predict_seq = lambda prefix: predict(prefix, 100, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        train_ppl, elapsed_time = train_epoch(net, train_iter, loss, optimizer, device, vocab)
        train_results.append(train_ppl)

        val_ppl = evaluate(net, val_iter, loss, device, vocab)
        val_results.append(val_ppl)

        if logger is not None:
            logger.info(f'epoch {epoch} - val-ppl: {val_ppl:.1f}, train-ppl: {train_ppl:.1f}, {elapsed_time=:.2f}s')

        if (epoch + 1) % 10 == 0:
            vocab_itos = vocab[0]

            if logger is not None:
                logger.info(predict_seq(['<BOS>']))
                logger.info(predict_seq(['<BOS>', vocab_itos[21]]))
                logger.info(predict_seq(['<BOS>', choice(vocab_itos)]))
                logger.info(predict_seq(['<BOS>', choice(vocab_itos)]))

            # plt results
            plt.plot(train_results, label='train')
            plt.xlabel('epoch')
            plt.ylabel('perplexity')
            plt.plot(val_results, label='val')
            plt.legend()
            plt.savefig(f"plots/{experiment_name}_ep{epoch + 1}.png")
            plt.clf()

    # inference after training
    logger.info(predict_seq(['<BOS>']))
    logger.info(f"train results (ppl):\n{train_results}")
    logger.info(f"val results (ppl):\n{val_results}")

    # plt results
    plt.plot(train_results, label='train')
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.plot(val_results, label='val')
    plt.legend()
    plt.savefig(f"plots/{experiment_name}_done.png")
    plt.clf()


def train_epoch(net, train_iter, loss, updater, device, vocab):
    """Train a net within one epoch"""
    start_time = time.time()
    metric = Accumulator(2)  # Sum of training loss, no. of tokens

    for batch in train_iter:
        seq = batch.content[0]
        X = seq[:, :-1]
        Y = seq[:, 1:]

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, _ = net(X)
        l = loss(y_hat, y.long()).mean()

        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

        # this way of computing the perplexity works only because the cross-entropy loss is used!
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), time.time() - start_time


def run_experiment(experiment_name: str, model: str = 'rnn_torch', tokenization: str = 'subword',
                   epochs: int = 500,
                   num_hiddens: int = 64,
                   level_type: str = ''):
    project_dir = pathlib.Path(os.path.abspath(__file__)).parent
    pathlib.Path.mkdir(project_dir / 'plots', exist_ok=True)
    csv_dir = project_dir / 'dataset'
    project_dir, csv_dir = str(project_dir), str(csv_dir)

    logger = create_logger(project_dir, experiment_name)
    logger.info(f"Starting run {experiment_name} with {model=} "
                f"{tokenization=} {epochs=} {num_hiddens=} device={get_device()}")

    models = {
        'rnn_scratch': RNNModelScratch,
        'lstm': LSTM,
        'stacked_lstm': StackedLstm,
        'rnn_torch': RNNModelPyTorch,
        'gru': GRU
    }

    batch_size = 64

    train_data, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('training', csv_dir,
                                                                       tokenization=tokenization,
                                                                       level_type=level_type)
    val_data, _, _, _ = brewed_dataLoader('validation', csv_dir, tokenization=tokenization,
                                          level_type=level_type)
    vocab = vocab_itos, vocab_stoi, vocab_size

    train_iter = torchtext.data.BucketIterator(train_data,
                                               batch_size=batch_size,
                                               sort_key=lambda x: len(x.content),
                                               sort_within_batch=True)
    val_iter = torchtext.data.BucketIterator(val_data,
                                             batch_size=16,
                                             sort_key=lambda x: len(x.content),
                                             sort_within_batch=True)

    architecture = models[model]
    net = architecture(vocab_size, num_hiddens, get_device())

    lr = 0.001
    train(net, train_iter, val_iter, vocab, lr, epochs, get_device(), logger, experiment_name)

    torch.save(net.state_dict(), f"{project_dir}/checkpoints/{experiment_name}_ep{epochs}.ckpt")

    logger.info(f"------------------")


if __name__ == '__main__':
    # run_experiment('gru-subword', 'gru', epochs=100, num_hiddens=128)
    # run_experiment('rnn_torch-subword', 'rnn_torch', epochs=100, num_hiddens=128)
    # run_experiment('rnn_scr-subword', 'rnn_scratch', epochs=100, num_hiddens=128)
    # run_experiment('rnn_scr-subword-wordLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
    #                level_type='wordLevel')
    run_experiment('rnn_scr-subword-byteLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='byteLevel')
    # run_experiment('rnn_scr-subword-charLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
    #                level_type='charLevel')
    # run_experiment('lstm-subword', 'lstm', epochs=100, num_hiddens=128)
    # run_experiment('lstm_stacked-subword', 'stacked_lstm', epochs=100, num_hiddens=128)

    # Subword-based BertTokenizer (pre-trained)
    # run_experiment('rnn_torch-subword-Bert', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword')
    # Word-Level BertTokenizer from Scratch
    # run_experiment('rnn_torch-subword-wordLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword', level_type='wordLevel')
    # Byte-Level BertTokenizer from Scratch
    # run_experiment('rnn_torch-subword-byteLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword', level_type='byteLevel')
    # Char-Level BertTokenizer from Scratch
    # run_experiment('rnn_torch-subword-charLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword', level_type='charLevel')
    # Sentence-Level BertTokenizer from Scratch
    # run_experiment('rnn_torch-subword-sentenceLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword', level_type='sentenceLevel')
    # Char-Level Tokenizer
    # run_experiment('rnn_torch-char', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='char')
    # Word-Level Tokenizer
    # run_experiment('rnn_torch-word', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='word')

    # run_experiment('rnn_scr-char', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='char')
    # run_experiment('lstm-char', 'lstm', epochs=100, num_hiddens=128, tokenization='char')

    # run_experiment('lstm_stacked-char', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='char')
    # run_experiment('rnn_torch-word', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='word')
    # run_experiment('rnn_scr-word', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='word')
    # run_experiment('lstm-word', 'lstm', epochs=100, num_hiddens=128, tokenization='word')
    # run_experiment('lstm_stacked-word', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='word')
