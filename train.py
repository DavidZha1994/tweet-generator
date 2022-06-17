import time

import torch  # pytorch == 1.11.0
import torch.nn as nn
import timeit
import torchtext
import logging
import random
from tqdm import tqdm
from transformers import BertTokenizer
import os
import pathlib

from models import TweetGenerator, RNNModelScratch, StackedLstm
from utils import Accumulator
import math
import matplotlib.pyplot as plt


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_model_idx(log_dir, model_name):
    current_cnt = 1
    for p in (pathlib.Path(log_dir) / 'log').iterdir():
        if str(p.name).startswith(model_name): current_cnt += 1
    return current_cnt


def gen_log(log_dir, file_name, logger):
    """
    Generate a logging file and add to logger

    :return: none
    """

    model_name = file_name
    (pathlib.Path(log_dir) / 'log').mkdir(exist_ok=True)
    idx = get_model_idx(log_dir, file_name)
    file_name += f'{idx:03d}'
    print("log file generated: " + file_name)
    file = logging.FileHandler(log_dir + "/log/" + file_name + ".log")
    file.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(file)
    logger.info(f"[Model name] {model_name}")
    return logger


def train(model, training_data, val_data, vocab_size, vocab_stoi, vocab_itos, optimizer, batch_size=1, num_epochs=1,
          lr=0.001,
          print_every=100, logger=None):
    train_results = []
    val_results = []

    criterion = nn.CrossEntropyLoss()
    data_iter = torchtext.data.BucketIterator(training_data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.content),
                                              sort_within_batch=True)
    val_iter = torchtext.data.BucketIterator(val_data,
                                             batch_size=16,
                                             sort_key=lambda x: len(x.content),
                                             sort_within_batch=True)

    logger.info(
        f"[Hyperparameter Settings] batch_size={batch_size} num_epochs={num_epochs} step_size={lr:.2f} iter={print_every}")

    for e in tqdm(range(num_epochs), desc='epoch'):
        metric = Accumulator(2)  # Sum of training loss, no. of tokens

        for (tweet, lengths), label in tqdm(data_iter, desc='training'):
            target = tweet[:, 1:].to(get_device())
            inp = tweet[:, :-1].to(get_device())

            # forward pass
            output, state = model(inp)
            loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1)).mean()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(model, 1)
            optimizer.step()

            metric.add(loss * output.numel(), output.numel())

        val_loss, val_ppl = evaluate(model, val_iter, criterion, get_device())
        train_ppl = math.exp(metric[0] / metric[1])
        print(f"epoch: {e}, val-loss: {val_loss}, val-ppl: {val_ppl}")
        logger.info(f"epoch: {e}, val-loss: {val_loss}, val-ppl: {val_ppl}, train-ppl: {train_ppl}")
        sample_seq = sample_sequence(model, vocab_stoi, vocab_itos, 140, 0.8)
        logger.info(f"[Generated Sequence] {sample_seq}")
        print(f"[Generated Sequence] {sample_seq}")

        print(f"train-ppl: {train_ppl}")
        train_results.append(train_ppl)
        val_results.append(val_ppl)

    return train_results, val_results


def grad_clipping(net, theta):
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


@torch.no_grad()
def evaluate(net, val_iter, criterion, device):
    metric = Accumulator(2)
    avg_loss = 0

    for (val_tweet, _), _ in val_iter:
        val_target = val_tweet[:, 1:].to(device)
        val_inp = val_tweet[:, :-1].to(device)
        y_hat, _ = net(val_inp)

        l = criterion(y_hat.reshape(-1, vocab_size), val_target.reshape(-1).long()).mean()
        avg_loss += l

        metric.add(l * val_target.numel(), val_target.numel())

    return avg_loss, math.exp(metric[0] / metric[1])


def train_logged(model, training_iter, val_iter, vocab_size, vocab_stoi, vocab_itos, optimizer, batch=1, epochs=1,
                 iterations=100, logger=None, project_dir=""):
    print('[Starting]')
    start = timeit.default_timer()
    train_res, val_res = train(model, training_iter, val_iter, vocab_size, vocab_stoi, vocab_itos, optimizer, batch,
                               epochs, lr, iterations, logger)

    stop = timeit.default_timer()
    print('[Done]')
    print(f'[Runtime] {float((stop - start) / 60):.2f}')
    logger.info(f"[Runtime] {float((stop - start) / 60)}")

    # Save the model checkpoint
    torch.save(model.state_dict(),
               project_dir + '/' + '[' + random.random().__str__() + ']'
               + 'RNN.ckpt')
    print('model saved.done')

    return train_res, val_res


def sample_sequence(model, vocab_stoi, vocab_itos, max_len=100, temperature=0.8):
    generated_sequence = ""
    # one-to-many
    inp = torch.Tensor([vocab_stoi["<BOS>"]]).long().to(get_device())
    # hidden = model.begin_state(inp.shape[0], get_device())
    for p in range(max_len):
        output, hidden = model(inp.unsqueeze(0))  # , hidden
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted character to string and use as next input
        predicted_char = vocab_itos[top_i]

        if predicted_char == "<EOS>":
            break
        generated_sequence += predicted_char
        inp = torch.Tensor([top_i]).long().to(get_device())
    return generated_sequence


def brewed_dataLoader(which_data, data_dir):  # which_ds could be 'training', 'validation'

    # Subword-based tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Character-based tokenization
    # tokenize = lambda x:x
    # Word-based tokenization
    # tokenize = lambda x: x.split()

    # it is for character/word-based tokenization
    text_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=tokenizer.tokenize,  # because are building a character/subword/word-RNN
                                      include_lengths=True,  # to track the length of sequences, for batching
                                      batch_first=True,
                                      use_vocab=True,  # to turn each character/word/subword into an integer index
                                      init_token="<BOS>",  # BOS token
                                      eos_token="<EOS>",  # EOS token
                                      unk_token=None)

    train_data, val_data = torchtext.data.TabularDataset.splits(
        path=data_dir,
        train='train_cleaned.csv',
        validation='val_cleaned.csv',
        format='csv',
        skip_header=True,
        fields=[
            ('', None),  # first column is unnamed
            ('content', text_field)
        ])

    text_field.build_vocab(train_data, val_data)
    vocab_stoi = text_field.vocab.stoi
    vocab_itos = text_field.vocab.itos
    vocab_size = len(text_field.vocab.itos)

    if which_data == 'validation':
        data = val_data
    else:
        data = train_data

    print("tweets content: ", data.examples[6].content)
    print("tweets length: ", len(data))
    print("vocab_size: ", vocab_size)

    return data, vocab_stoi, vocab_itos, vocab_size


def plot_train_val(train_results, val_results, metric='perplexity', scale='linear', title=""):
    plt.plot(train_results, label='train')
    plt.xlabel('epoch')
    plt.yscale(scale)
    plt.ylabel(metric)
    plt.plot(val_results, label='val')
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    num_hiddens = 128
    batch_size = 64
    num_epochs = 100
    lr = 0.001
    iterations = 800

    logger = logging.getLogger(__name__)
    project_dir = pathlib.Path(os.path.abspath(__file__)).parent
    csv_dir = project_dir / 'dataset'
    project_dir, csv_dir = str(project_dir), str(csv_dir)

    train_data, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('training', csv_dir)
    val_data, _, _, _ = brewed_dataLoader('validation', csv_dir)
    # testing_data, testing_iter, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('testing')

    models = [TweetGenerator(vocab_size, hidden_size=num_hiddens, device=get_device()),
              RNNModelScratch(vocab_size, num_hiddens, device=get_device()),
              StackedLstm(vocab_size, num_hiddens, device=get_device())]

    for i in range(len(models)):
        optimizer = torch.optim.Adam(models[i].parameters(), lr=lr)

        model_name = models[i].__class__.__name__
        gen_log(project_dir, model_name, logger)
        train_res, val_res = train_logged(models[i], train_data, val_data, vocab_size, vocab_stoi, vocab_itos, optimizer,
                                          batch_size, num_epochs, iterations, logger, project_dir)

        plot_title = f"{model_name}{get_model_idx(project_dir+'/', model_name)-1:03d}"
        plot_train_val(train_res, val_res, scale='linear', title=plot_title)
        plot_train_val(train_res, val_res, scale='log', title=plot_title)
