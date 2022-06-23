import random
import collections
import re
import os
from tokenize import Whitespace

import torch
import logging
import pandas as pd

from tokenizers.implementations import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, \
    BertWordPieceTokenizer
from transformers import BertTokenizer
from tokenizers import Tokenizer, pre_tokenizers, trainers, models
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/dataset/gatsby.txt'


def read_great_gatsby():
    """Load great gatsby text"""
    with open('dataset/gatsby.txt', 'r', encoding="utf8") as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens by using the pre-trained tokenizer."""

    if token == 'word':
        print(f"[Tokenizer] {token}")
        return [line.split() for line in lines]
    elif token == 'char':
        print(f"[Tokenizer] {token}")
        return [list(line) for line in lines]
    elif token == 'subword':
        print(f"[Tokenizer] {token}")
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        return [tok.tokenize(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def get_training_corpus():
    dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]

    with open("wikitext-2.txt", "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            f.write(dataset[i]["text"] + "\n")


def train_tokenizer():
    """Train a tokenizer from scratch"""
    # Initialize an empty tokenizer from
    # ByteLevelBPETokenizer/ CharBPETokenizer/ BertWordPieceTokenizer/ SentencePieceBPETokenizer
    tokenizer = CharBPETokenizer()
    tokenizer.train(files='./dataset/gatsby.txt',
                    vocab_size=20000,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                    ])
    tokenizer.save('./my_token/' + type(tokenizer).__name__ + '.json')
    return tokenizer


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_gatsby(start_token=0, max_tokens=-1, relative_size=False):
    """Return token indices and the vocabulary of the gatsby machine dataset."""
    lines = read_great_gatsby()
    tokens = tokenize(lines, 'subword')
    vocab = Vocab(tokens)
    # Since each text line in the gatsby dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if not relative_size:
        corpus = corpus[start_token:max_tokens]
    else:
        corp_size = len(corpus)
        start = int(start_token * corp_size)
        end = int(max_tokens * corp_size)
        corpus = corpus[start:end]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """An iterator to load sequence data."""

    def __init__(self, batch_size, num_steps, use_random_iter, start_token, max_tokens, relative_size):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_gatsby(start_token, max_tokens, relative_size)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_gatsby(batch_size, num_steps,
                     use_random_iter=False, start_token=0, max_tokens=10000, relative_size=False):
    """Return the iterator and the vocabulary of the gatsby dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, start_token, max_tokens, relative_size)
    return data_iter, data_iter.vocab


def load_data_mask():
    # TODO: put the dataLoader of mask's tweets
    pass


def load_data_trump():
    # TODO: put the dataLoader of trump's tweets
    pass


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def gen_log(dir_to_save=ROOT_DIR):
    mylogs = logging.getLogger(__name__)
    mylogs.setLevel(logging.INFO)

    log_name = '[' + random.randint(0, 100).__str__() + ']' + 'RNN.log'
    print(f"log file \'{log_name}\' is generated. ")
    file = logging.FileHandler(dir_to_save + "/log/" + log_name)
    file.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
    file.setFormatter(file_format)
    mylogs.addHandler(file)
    return mylogs


def csv2txt(csv_dir, colname):
    df = pd.read_csv(csv_dir, skip_blank_lines=True)
    df[colname].to_csv('./dataset/csv2seq.txt', sep="\n", index=False, header=False)


if __name__ == "__main__":
    # ByteLevelBPETokenizer/ CharBPETokenizer/ BertWordPieceTokenizer/ SentencePieceBPETokenizer
    # tok = Tokenizer.from_file('./my_token/ByteLevelBPETokenizer.json')
    # res = tok.encode("I made the mistake of using the tokenizers library with a ByteLevelBPETokenizer, "
    #                  "which uses the 0th and 1st for '!' and '' no matter what I do. ")
    # print(res.tokens)
    csv2txt('./dataset/train_cleaned.csv', 'content')
