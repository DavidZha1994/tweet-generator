import random
import collections
import re
import os
import torch
import logging
import pandas as pd
# import preprocessor as pp  # tweet-preprocessor == 0.6.0
import nltk
import json

import torchtext
from nltk.probability import FreqDist
from tokenizers.implementations import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, \
    BertWordPieceTokenizer
from transformers import BertTokenizer
from tokenizers import Tokenizer
from transformers import BertTokenizer
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def train_tokenizer(training_source_dir='./dataset/gatsby.txt'):
    """Train a tokenizer from scratch"""
    # Initialize an empty tokenizer from
    # ByteLevelBPETokenizer/ CharBPETokenizer/ BertWordPieceTokenizer/ SentencePieceBPETokenizer
    tokenizer = CharBPETokenizer()
    tokenizer.train(files=training_source_dir,
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


def csv2txt(csv_dir, txt_dir, col_name='content'):
    df = pd.read_csv(csv_dir, skip_blank_lines=True)
    df[col_name].to_csv(txt_dir, sep="\n", index=False, header=False)


# def preprocessing(csv_dir, cleaned_csv_dir, col_name='tweet'):
#     tweets_csv = pd.read_csv(csv_dir)  # read the original csv. file.
#     df = pd.DataFrame(tweets_csv)  # convert csv. file into DataFrame with a 2 dimensional data structure.
#
#     cleaned_tweets_list = [pp.clean(content) for content in df[col_name] if
#                            pp.clean(content) != '']  # clean the content by using package 'preprocessor'.
#
#     cleaned_tweets_dict = {'content': cleaned_tweets_list}  # create a dictionary to save cleaned data.
#     cleaned_tweets_df = pd.DataFrame(cleaned_tweets_dict)  # convert dictionary into DataFrame
#     cleaned_tweets_df.to_csv(cleaned_csv_dir)  # convert DataFrame into csv. file and save in the same path.
#
#     return cleaned_tweets_df, cleaned_tweets_dict


def token_counter(tokenizer_dir='./my_token/CharBPETokenizer_Musk_cleaned.json',
                  csv_dir='./dataset/val_cleaned.csv',
                  json_dir='./my_token/val_token_freq.json'):
    trained_tokenizer = Tokenizer.from_file(tokenizer_dir)
    df = pd.read_csv(csv_dir)

    token_list = []
    for row in df.iterrows():
        encode = trained_tokenizer.encode(row[1]['content'])  # 'content' can also be other column names.
        token_list.extend(encode.tokens)

    token_fd = nltk.FreqDist(token_list)
    json_dict = {'token_frequency': []}

    with open(json_dir, 'w') as f:
        for key in token_fd:
            json_dict['token_frequency'].append({key: token_fd[key]})
        json_str = json.dumps(json_dict)
        f.write(json_str)
    f.close()

    return token_fd


def token_freq_diff(json1_dir, json2_dir):
    """This method is used to check whether the train contains the token of the test, and it returns a json file
    containing all the tokens that do not exist in the train."""
    with open(json1_dir, "r") as f1:
        json1 = json.loads(f1.read())
    with open(json2_dir, "r") as f2:
        json2 = json.loads(f2.read())

    diff_dict = {'token_frequency_diff': []}
    in_list = False
    with open('my_token/token_frequency_diff.json', 'w') as f:
        for item2 in json2['token_frequency']:  # test token freq
            for item1 in json1['token_frequency']:  # training token freq
                if item2.keys() == item1.keys():
                    in_list = True
                    break
            if not in_list:
                key = list(item2.keys())[0]
                value = list(item2.values())[0]
                diff_dict['token_frequency_diff'].append({key: value})

        json_str = json.dumps(diff_dict)
        f.write(json_str)
    f.close()


def brewed_dataLoader(which_data, data_dir, tokenization='char', level_type=''):  # which_ds could be 'training', 'validation'
    """
    Note that there are two steps should be done at the beginning if the developer wants to use subword-based tokenization:
    (1) Create an instance of tokenizer.
        (option 1) Use pre-trained tokenizer of Huggingface ('hf'):
        hf_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        (option 2) Use homemade ('hm') pre-trained tokenizer:
        hm_tokenizer = Tokenizer.from_file(PROJECT_DIR + '/my_token/CharBPETokenizer_Musk_cleaned.json')
    (2) Adapt the parameter 'tokenize' in torchtext.data.Field(...).
        (option 1) For the instance 'hf_tokenizer':
        tokenize = hf_tokenizer.tokenize
        (option 2) For the instance 'hm_tokenizer':
        tokenize = lambda x: hm_tokenizer.encode(x).tokens
    Otherwise, there is no need to create an instance of tokenizer if the developer only wants to use character/word-based tokenization:
        (character-based) tokenize = lambda x:x
        (word-based) tokenize = lambda x: x.split()
    """

    # Subword-based tokenization
    # tokenize = BertTokenizer.from_pretrained("bert-base-uncased").tokenize
    # Character-based tokenization
    # tokenize = lambda x:x
    # Word-based tokenization
    tokenize = lambda x: x.split()

    if tokenization == 'char':
        # character level tokenization
        tokenize = lambda x: x
    elif tokenization == 'word':
        # word level tokenization
        tokenize = lambda x: x.split()
    elif tokenization == 'subword':
        # sub-word level tokenization
        if level_type == 'wordLevel':
            brewed_tokenizer = Tokenizer.from_file('./my_token/BertWordPieceTokenizer_train.json')
            tokenize = lambda x: brewed_tokenizer.encode(x).tokens
        elif level_type == 'byteLevel':
            brewed_tokenizer = Tokenizer.from_file('./my_token/ByteLevelBPETokenizer_train.json')
            tokenize = lambda x: brewed_tokenizer.encode(x).tokens
        elif level_type == 'charLevel':
            brewed_tokenizer = Tokenizer.from_file('./my_token/CharBPETokenizer_train.json')
            tokenize = lambda x: brewed_tokenizer.encode(x).tokens
        elif level_type == 'sentenceLevel':
            brewed_tokenizer = Tokenizer.from_file('./my_token/SentencePieceBPETokenizer_train.json')
            tokenize = lambda x: brewed_tokenizer.encode(x).tokens
        else:
            tokenize = BertTokenizer.from_pretrained("bert-base-uncased").tokenize
    else:
        raise Exception(
            "Wrong parameter for 'tokenization'-argument please use one of these: 'char', 'word', 'subword'")

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


if __name__ == "__main__":
    # ByteLevelBPETokenizer/ CharBPETokenizer/ BertWordPieceTokenizer/ SentencePieceBPETokenizer
    # train_tokenizer('./dataset/combined_Musks_tweets_cleaned.txt')
    # tok = Tokenizer.from_file('./my_token/CharBPETokenizer_Musk_cleaned.json')
    # res = tok.encode("Vaccines are just the start. Its also capable in theory of curing almost anything. "
    #                  "fukushima Turns medicine into a software &amp; simulation problem.")
    # print(res.ids)
    # res_ids = res.ids
    # print(tok.decode(res_ids))
    # preprocessing('./dataset/combined_Musks_tweets.csv', './dataset/combined_Musks_tweets_cleaned.csv')
    # csv2txt('./dataset/combined_Musks_tweets_cleaned.csv', './dataset/combined_Musks_tweets_cleaned.txt', 'content')
    # token_counter(csv_dir='./dataset/realdonaldtrump_cleaned.csv',
    #               json_dir='./my_token/realdonaldtrump_token_freq.json')
    # token_freq_diff('./my_token/training_token_freq.json', './my_token/val_token_freq.json')

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(
        bert_tokenizer.tokenize("Vaccines are just the start. Its also capable in theory of curing almost anything. "))

    brewed_tokenizer = Tokenizer.from_file('./my_token/CharBPETokenizer_Musk_cleaned.json')
    print(
        brewed_tokenizer.encode(
            "Vaccines are just the start. Its also capable in theory of curing almost anything. ").tokens)
