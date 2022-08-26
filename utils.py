import random
import collections
import re

from transformers import BertTokenizer, GPT2Tokenizer
import torchtext

import torch


def read_great_gatsby():
    """Load great gatsby text"""
    with open('dataset/gatsby.txt', 'r', encoding="utf8") as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens."""

    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


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
    tokens = tokenize(lines, 'char')
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


def get_tokenization_fn(tokenization='char'):
    if tokenization == 'char':
        # character level tokenization
        tokenize = lambda x: x
    elif tokenization == 'word':
        # word level tokenization
        # first the string is split at spaces and some special characters, afterwards "' characters are removed from each token
        # and lastly empty characters '' are removed.
        tokenize = lambda x: list(filter(lambda z: z != '', map(lambda y: re.sub("[\"\']", '', y),
                                 re.split("(?=&amp;)|(?=[()!.?,$~+*])|(?<=[()!.?,$~+*])|\d| ", x.removesuffix("\n")))))
    elif tokenization == 'subword':
        # sub-word level tokenization
        tokenize = BertTokenizer.from_pretrained("bert-base-uncased").tokenize
    elif tokenization == 'gpt2-trained':
        tokenizer = GPT2Tokenizer('gpt2tokenizer/vocab.json', 'gpt2tokenizer/merges.txt')
        tokenizer.pad_token = tokenizer.eos_token
        tokenize = tokenizer.tokenize
    elif tokenization == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenize = tokenizer.tokenize
    else:
        raise Exception(
            "Wrong parameter for 'tokenization'-argument please use one of these: 'char', 'word', 'subword'")

    return tokenize


def brewed_dataLoader(which_data, data_dir, tokenization='char'):  # which_ds could be 'training', 'validation'

    tokenize = get_tokenization_fn(tokenization)


    # it is for character/word-based tokenization
    text_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=tokenize,  # because are building a character/subword/word-RNN
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


def get_prompts():
    prompts = set()
    with open("./dataset/combined_Musk_tweets_cleaned.txt") as file:
        for line in file:
            beginning = line.split(' ')[0]

            beginning = beginning.replace('\n', '')
            beginning = beginning.replace('\\', '')
            beginning = beginning.replace('.', '')
            beginning = beginning.replace('!', '')
            beginning = beginning.replace('\"', '')
            if len(beginning) > 1:
                prompts.add(beginning)

    return sorted(list(prompts))


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
