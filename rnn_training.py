import math
import time

import torch
import re
from torch import nn
import logging
from utils import Accumulator, brewed_dataLoader, get_device, brewed_dataLoader_encodings
import matplotlib.pyplot as plt
import os
import sys
import pathlib
import torchtext
from random import choice
from models import StackedLstm, VanillaRNN, LSTM, GRU, TweetGenerator

is_logger_adjusted = False
TOKENIZER_TYPE = 'char'

def predict(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`."""

    # unpack the vocab to use
    vocab_itos, vocab_stoi, _ = vocab
    state = None
    outputs = [vocab_stoi[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # Warm-up period
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab_stoi[y])
    # Generate 'num_preds' tokens after the warm up period
    for _ in range(num_preds):
        # do the forward pass through the network
        y, state = net(get_input(), state)
        # add predicted token
        outputs.append(int(y.argmax(dim=1).reshape(1)))

        if outputs[-1] == vocab_stoi['<EOS>']:
            break

    # create a single string from the generated tokens
    return ' '.join([vocab_itos[i] for i in outputs])


@torch.no_grad()
def evaluate(net, val_iter, loss, device, vocab):
    """
    Evaluate the net (e.g. with a validation data set).

    :param net: net to evaluate
    :param val_iter: iterator holding the data set
    :param loss: loss function that should be used
    :param device: device to run the computations on (cpu or cuda:x)
    :param vocab: vocabulary for the network
    :return: tuple of token level perplexity and word level perplexity
    """

    # init accumulators to store entropies for later conversion to perplexity
    metric_emb = Accumulator(2)  # one for the token level
    metric_words = Accumulator(2)  # one for the word level

    # do the actual forward pass for each batch
    for batch in val_iter:
        seq = batch.content[0]
        X = seq[:, :-1]
        Y = seq[:, 1:]

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, _ = net(X)
        l = loss(y_hat.reshape(-1, len(vocab[0])), y.long()).mean()

        metric_emb.add(l * y.numel(), y.numel())

        word_counts = [count_words(X[i], TOKENIZER_TYPE, vocab) for i in range(X.shape[0])]
        words_in_batch = sum(word_counts)
        metric_words.add(l * words_in_batch, words_in_batch)

    return math.exp(metric_emb[0] / metric_emb[1]), math.exp(metric_words[0] / metric_words[1])


def grad_clipping(net, theta):
    """Clip the gradient to prevent exploding gradients."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def count_words(encoding, tokenization, vocab):
    # get the type of separating element
    vocab_itos = vocab[0]
    vocab_stoi = vocab[1]
    if tokenization == 'char':
        # count the number of spaces and add 1 for the number of words
        return [i.item() for i in encoding].count(vocab_stoi[' ']) + 1
    if tokenization == 'word':
        # count the number of words directly (leaving out the <BOS> token)
        return len(encoding) - 1
    if tokenization == 'gpt2-trained' or tokenization == 'gpt2':
        # at the beginning of each new word is a 'Ġ'-character, so we can count these
        decoding = [vocab_itos[i.item()] for i in encoding]
        return ''.join(decoding).count("\u0120") - 1
    if tokenization == 'subword':
        # subwords are concatenated if they begin with ##, so we can subtract these from the sentence length
        decoding = [vocab_itos[i.item()] for i in encoding]
        decoding = ''.join(decoding)
        return len(encoding) - len(re.findall("(##)", decoding)) - 1


def train_epoch(net, train_iter, loss, updater, device, vocab):
    """
    Train a net for one epoch.

    :param net: network to train.
    :param train_iter: training iterator holding the training data.
    :param loss: loss function. Use cross-entropy loss to retrieve reasonable perplexity values.
    :param updater: torch updater for the back propagation.
    :param device: Device to run the computations on (cpu or cuda:x)
    :param vocab: vocab used with the tokenizer.
    :return: triple of perplexity on token level, perplexity on word level, runtime
    """
    start_time = time.time()
    metric_embedding = Accumulator(2)  # accumulator for loss normalized by the sequence length on token level
    metric_words = Accumulator(2)  # accumulator for loss normalized by the sequence length on word level

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

        # used to calculate perplexity: normalization based on the number of tokens
        metric_embedding.add(l * y.numel(), y.numel())

        # second way to calculate perplexity: normalization based on the number of words (dependent on tokenizer)
        # TODO this way of counting words is quite inefficient
        word_counts = [count_words(X[i], TOKENIZER_TYPE, vocab) for i in range(X.shape[0])]
        words_in_batch = sum(word_counts)
        metric_words.add(l * words_in_batch, words_in_batch)

    # this way of computing the perplexity works only because the cross-entropy loss is used!
    ppl_emb = math.exp(metric_embedding[0] / metric_embedding[1])
    ppl_word = math.exp(metric_words[0] / metric_words[1])
    return ppl_emb, ppl_word, time.time() - start_time


def train(net, train_iter, val_iter, vocab, lr, num_epochs, device, logger=None, experiment_name='experiment', metric='word'):
    """Train a model"""
    loss = nn.CrossEntropyLoss()
    train_results = []
    val_results = []

    # Initialize
    optimizer = torch.optim.Adam(net.parameters(), lr)

    predict_seq = lambda prefix: predict(prefix, 100, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        train_ppl_emb, train_ppl_words, elapsed_time = train_epoch(net, train_iter, loss, optimizer, device, vocab)
        train_ppl = train_ppl_words if metric == 'word' else train_ppl_emb

        train_results.append(train_ppl)

        val_ppl_emb, val_ppl_words = evaluate(net, val_iter, loss, device, vocab)
        val_ppl = val_ppl_words if metric == 'word' else val_ppl_emb
        val_results.append(val_ppl)

        if logger is not None:
            logger.info(f'epoch {epoch} - val-ppl: {val_ppl:.1f}, train-ppl: {train_ppl:.1f}, {elapsed_time=:.2f}s')

        if (epoch + 1) % 10 == 0:
            vocab_itos = vocab[0]

            if logger is not None:
                # generate some sample sequences
                logger.info(predict_seq(['<BOS>']))
                logger.info(predict_seq(['<BOS>', vocab_itos[21]]))
                logger.info(predict_seq(['<BOS>', vocab_itos[22]]))
                logger.info(predict_seq(['<BOS>', choice(vocab_itos)]))
                logger.info(predict_seq(['<BOS>', choice(vocab_itos)]))
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


def create_logger(project_dir: str, file_name: str):
    """
    Generate a logging file and add to logger
    :return: logger
    """

    (pathlib.Path(project_dir) / 'log').mkdir(exist_ok=True)
    print("log file generated: " + file_name)
    file = logging.FileHandler(project_dir + "/log/" + file_name + ".log", encoding='utf-8')
    file.setLevel(logging.INFO)
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file)

    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


def run_experiment(experiment_name: str, model: str = 'lstm', tokenization: str = 'subword', epochs: int = 500,
                   num_hiddens: int = 64, metric='word'):
    """
    Run an experiment (i.e. the training and validation process) for a model-tokenizer combination.

    :param experiment_name: The name of the experiment which is also the name of the generated log file.
    :param model: model to use for the experiment. Options are: 'lstm', 'gru', 'stacked_lstm', 'rnn_scratch' (vanilla rnn)
    :param tokenization: tokenizer to use for the experiment. Options are 'char', 'word, 'gpt2', 'gpt2-trained'.
    :param epochs: number of epochs to train for.
    :param num_hiddens: number of hidden units for the network.
    :param metric: If set to word the perplexity is normalized with the word count, otherwise with the token count.
    :return: None
    """
    # necessary to get the tokenization when doing the evaluation
    global TOKENIZER_TYPE
    TOKENIZER_TYPE = tokenization

    project_dir = pathlib.Path(os.path.abspath(__file__)).parent
    pathlib.Path.mkdir(project_dir / 'plots', exist_ok=True)
    csv_dir = project_dir / 'dataset'
    project_dir, csv_dir = str(project_dir), str(csv_dir)

    logger = create_logger(project_dir, experiment_name)
    logger.info(f"Starting run {experiment_name} with {model=} "
                f"{tokenization=} {epochs=} {num_hiddens=} device={get_device()}")

    models = {
        'rnn_scratch': VanillaRNN,
        'lstm': LSTM,
        'stacked_lstm': StackedLstm,
        'rnn_torch': TweetGenerator,
        'gru': GRU
    }

    batch_size = 64

    train_data, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('training', csv_dir, tokenization=tokenization)
    val_data, _, _, _ = brewed_dataLoader('validation', csv_dir, tokenization=tokenization)
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
    train(net, train_iter, val_iter, vocab, lr, epochs, get_device(), logger, experiment_name, metric)

    torch.save(net.state_dict(), f"{project_dir}/checkpoints/{experiment_name}_ep{epochs}.ckpt")

    logger.info(f"------------------")


def run_experiment_encodings(experiment_name: str, model: str = 'rnn_torch', tokenization: str = 'subword',
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
        'rnn_scratch': VanillaRNN,
        'lstm': LSTM,
        'stacked_lstm': StackedLstm,
        'rnn_torch': TweetGenerator,
        'gru': GRU
    }

    batch_size = 64

    train_data, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader_encodings('training', csv_dir,
                                                                       tokenization=tokenization,
                                                                       level_type=level_type)
    val_data, _, _, _ = brewed_dataLoader_encodings('validation', csv_dir, tokenization=tokenization,
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
    # train all model-tokenizer combinations for the RNNs
    run_experiment('gru-subword', 'gru', epochs=100, num_hiddens=128)
    run_experiment('rnn_scr-subword', 'rnn_scratch', epochs=100, num_hiddens=128)
    run_experiment('lstm-subword', 'lstm', epochs=100, num_hiddens=128)
    run_experiment('lstm_stacked-subword', 'stacked_lstm', epochs=100, num_hiddens=128)
    run_experiment('gru-char', 'gru', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('rnn_scr-char', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('lstm-char', 'lstm', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('lstm_stacked-char', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('gru-word-fixed', 'gru', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('rnn_scr-word-fixed', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('lstm-word-fixed', 'lstm', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('lstm_stacked-word-fixed', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('gru-gpt-word', 'gru', epochs=100, num_hiddens=128, tokenization='gpt2-trained')
    run_experiment('rnn_scr-gpt-word', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='gpt2-trained')
    run_experiment('lstm-gpt-word', 'lstm', epochs=100, num_hiddens=128, tokenization='gpt2-trained')
    run_experiment('lstm_stacked-gpt-word', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='gpt2-trained')
    run_experiment('gru-gpttoken', 'gru', epochs=100, num_hiddens=128, tokenization='gpt2')
    run_experiment('rnn_scr-gpttoken', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='gpt2')
    run_experiment('lstm-gpttoken', 'lstm', epochs=100, num_hiddens=128, tokenization='gpt2')
    run_experiment('lstm_stacked-gpttoken', 'stacked_lstm', epochs=100, num_hiddens=128, tokenization='gpt2')
