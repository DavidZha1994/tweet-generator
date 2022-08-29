import pathlib
import random
import os
import torch
import logging
import pandas as pd
import preprocessor as pp  # tweet-preprocessor == 0.6.0
import nltk
import json

import torchtext
from tokenizers.implementations import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, \
    BertWordPieceTokenizer
from tokenizers import Tokenizer
from transformers import BertTokenizer, GPT2Tokenizer

from models import RNNModelScratch, RNNModelPyTorch
from train_rnn_torch import create_logger, get_device, train, run_experiment

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
nltk.download('punkt')


def csv2txt(csv_dir, txt_dir, col_name='content'):
    """Convert csv file to txt format"""
    df = pd.read_csv(csv_dir, skip_blank_lines=True)
    df[col_name].to_csv(txt_dir, sep="\n", index=False, header=False)


def gen_log(dir_to_save=ROOT_DIR):
    """"Create a logging file and add to logger"""
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


def load_ckpt_models(ckpt_name, net):
    """Load the pre-trained model"""
    ckpt_dir = pathlib.Path(os.path.abspath(__file__)).parent / 'checkpoints'
    ckpt = torch.load(ckpt_dir + ckpt_name + '.ckpt')
    return net.load_state_dict(ckpt)


def split_combi_dataset():
    """
        Split the data set into training - and test - and validation test sets
        with ratios 50% 25% 25%.
    """
    combined_csv = pd.read_csv('../dataset/realdonaldtrump_cleaned.csv')

    if not os.path.exists("../dataset/trump_combined_clean.csv"):
        combined_csv.to_csv("./dataset/trump_combined_clean.csv", index=False, encoding='utf-8')

    df = combined_csv[0:int(combined_csv.shape[0] / 2)]
    df.to_csv('./dataset/trump_train.csv', index=False)  # training dataset contains the first 50% of the whole set

    df = combined_csv[int(combined_csv.shape[0] / 2) + 1:int(combined_csv.shape[0] / 4 * 3)]
    df.to_csv('./dataset/trump_val.csv', index=False)  # validation dataset contains the 51% and 70% of the whole set

    df = combined_csv[int(combined_csv.shape[0] / 4 * 3) + 1:combined_csv.shape[0]]
    df.to_csv('./dataset/trump_test.csv', index=False)  # test dataset contains the 71% and 100% of the whole set


def preprocessing(csv_dir, cleaned_csv_dir, col_name='tweet'):
    """remove the undesired contents"""
    tweets_csv = pd.read_csv(csv_dir)  # read the original csv. file.
    df = pd.DataFrame(tweets_csv)  # convert csv. file into DataFrame with a 2 dimensional data structure.

    cleaned_tweets_list = [pp.clean(content) for content in df[col_name] if
                           pp.clean(content) != '']  # clean the content by using package 'preprocessor'.

    cleaned_tweets_dict = {'content': cleaned_tweets_list}  # create a dictionary to save cleaned data.
    cleaned_tweets_df = pd.DataFrame(cleaned_tweets_dict)  # convert dictionary into DataFrame
    cleaned_tweets_df.to_csv(cleaned_csv_dir)  # convert DataFrame into csv. file and save in the same path.

    return cleaned_tweets_df, cleaned_tweets_dict


def token_counter(tokenizer_dir='./my_token/CharBPETokenizer_Musk_cleaned.json',
                  csv_dir='./dataset/val_cleaned.csv',
                  json_dir='./my_token/val_token_freq.json'):
    """
    * This method is used to calculate the frequency of each token in the given dataset.
    * How it works?
        - The tokenizer and the dataset must be given.
        - The tokenizer will tokenize the dataset as tokens.
        - The frequency of each token will be calculated and then stored in a JSON file.
        - The JSON file will be returned.
    """
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


def gpt2_token_counter(tokenizer_dir='./my_token/CharBPETokenizer_Musk_cleaned.json',
                       gpt2_tokenizer=None,
                       csv_dir='./dataset/val_cleaned.csv',
                       json_dir='./my_token/val_token_freq.json'):
    """This method is used to calculate the frequency of each token in the given dataset."""
    if gpt2_tokenizer is None:
        trained_tokenizer = Tokenizer.from_file(tokenizer_dir)
    else:
        trained_tokenizer = gpt2_tokenizer
    df = pd.read_csv(csv_dir)

    token_list = []
    for row in df.iterrows():
        encode = trained_tokenizer.encode(row[1]['content'])  # 'content' can also be other column names.
        token_list.extend(trained_tokenizer.convert_ids_to_tokens(encode))

    token_fd = nltk.FreqDist(token_list)
    json_dict = {'token_frequency': []}

    with open(json_dir, 'w') as f:
        for key in token_fd:
            json_dict['token_frequency'].append({key: token_fd[key]})
        json_str = json.dumps(json_dict)
        f.write(json_str)
    f.close()

    return token_fd


def token_freq_diff(train_json_dir, test_json_dir, diff_name='tokens_not_in_train'):
    """
        This method is used to check whether the train contains the token of the test, and it returns a json file
        containing all the tokens that do not exist in the train.
    """
    dir = '../my_token/token_frequency/token_diff'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Directory ", dir, " Created ")
    else:
        print("Directory ", dir, " already exists")

    with open(train_json_dir, "r") as f1:
        json1 = json.loads(f1.read())
    with open(test_json_dir, "r") as f2:
        json2 = json.loads(f2.read())

    diff_dict = {diff_name: []}
    with open(f'{dir}/{diff_name}.json', 'w') as f:
        for item2 in json2['token_frequency']:  # test token freq
            in_list = False
            for item1 in json1['token_frequency']:  # training token freq
                if item2.keys() == item1.keys():
                    in_list = True  # test token is found in training set
                    break
            if not in_list:
                key = list(item2.keys())[0]
                value = list(item2.values())[0]
                diff_dict[diff_name].append({key: value})

        json_str = json.dumps(diff_dict)
        f.write(json_str)
    f.close()


def gen_log(dir_to_save=ROOT_DIR):
    """generate loggings"""
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


def get_vocab(tokenization='word'):
    """load dataset and retrieve the vocabulary"""
    csv_dir = pathlib.Path(os.path.abspath(__file__)).parent / 'dataset'
    _, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('validation', csv_dir, tokenization=tokenization,
                                                              level_type='')

    vocab = vocab_itos, vocab_stoi, vocab_size
    return vocab


def train_tokenizer(training_source_dir='./dataset/gatsby.txt', tokenizer_name='BERT', tokenizer_prefix=''):
    """
        Train a tokenizer from scratch and initialize an empty tokenizer
        from ByteLevelBPETokenizer/ BertWordPieceTokenizer/ SentencePieceBPETokenizer
    """
    init_tokenizer = {
        'CBPE': CharBPETokenizer(),
        'BPE': ByteLevelBPETokenizer(),
        'BERT': BertWordPieceTokenizer(),
        'SBPE': SentencePieceBPETokenizer()
    }
    tokenizer = init_tokenizer[tokenizer_name]
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
                        "[UNK]"]
                    )
    tokenizer.save(f'./my_token/{tokenizer_prefix}_{tokenizer_name}_tokenizer.json')
    # tokenizer.save_model('./my_token/tokenizers_scratch')
    return tokenizer


def train_gpt2_tokenizer(training_source_dir='./dataset/gatsby.txt', tokenizer_name='gpt2_tokenizer'):
    """
        Train a wordpiece tokenizer from scratch
    """
    dir = f'my_token/tokenizers_scratch/{tokenizer_name}/'
    init_tokenizer = ByteLevelBPETokenizer()
    tokenizer = init_tokenizer
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
                        "[UNK]"]
                    )

    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Directory ", dir, " Created ")
    else:
        print("Directory ", dir, " already exists")

    tokenizer.save_model(dir)
    tokenizer = GPT2Tokenizer(f'{dir}vocab.json',
                              f'{dir}merges.txt')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_vocabulary(dir)
    return tokenizer


def brewed_dataLoader(which_data, data_dir, tokenization='char',
                      level_type=''):  # which_ds could be 'training', 'validation'
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

    # inject data sets
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

    # build vocabulary from training- and validation data
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


def dataLoader_for_rq(which_data, data_dir, tokenizer_name='musk'):
    """
        load pre-trained tokenizers_scratch.
    """
    tokenizer = {
        'musk': Tokenizer.from_file('../my_token/tokenizers/musk_train_BERT_tokenizer.json'),
        'gpt2_musk': GPT2Tokenizer('../my_token/tokenizers/gpt2_musk_tokenizer/vocab.json',
                                   'my_token/tokenizers_scratch/gpt2_musk_tokenizer/merges.txt'),
        'trump': Tokenizer.from_file('../my_token/tokenizers/trump_train_BERT_tokenizer.json'),
        'gpt2_trump': GPT2Tokenizer('../my_token/tokenizers/gpt2_trump_tokenizer/vocab.json',
                                    'my_token/tokenizers_scratch/gpt2_trump_tokenizer/merges.txt'),
        'trump_combi': None

    }
    tokenizer = tokenizer[tokenizer_name]
    if tokenizer_name in {'gpt2_musk', 'gpt2_trump'}:
        tokenize = lambda x: tokenizer.convert_ids_to_tokens(tokenizer.encode(x))
    else:
        tokenize = lambda x: tokenizer.encode(x).tokens

    # it is for character/word-based tokenization
    text_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=tokenize,
                                      include_lengths=True,  # to track the length of sequences, for batching
                                      batch_first=True,
                                      use_vocab=True,  # to turn each character/word/subword into an integer index
                                      init_token="<BOS>",  # BOS token
                                      eos_token="<EOS>",  # EOS token
                                      unk_token=None)

    train_ds, val_ds = 'train_cleaned.csv', 'val_cleaned.csv'
    if tokenizer_name in {'trump', 'gpt2_trump'}:
        train_ds, val_ds = 'trump_train.csv', 'trump_val.csv'

    train_data, val_data = torchtext.data.TabularDataset.splits(
        path=data_dir,
        train=train_ds,
        validation=val_ds,
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


def run_experiment_for_rq(experiment_name: str, model: str = 'rnn_torch', tokenization: str = 'subword',
                          epochs: int = 500,
                          num_hiddens: int = 64,
                          tokenizer_name: str = 'musk'):
    """adapted experimental settings for tokenization analysis (research questions)"""
    project_dir = pathlib.Path(os.path.abspath(__file__)).parent
    pathlib.Path.mkdir(project_dir / 'plots', exist_ok=True)
    csv_dir = project_dir / 'dataset'
    project_dir, csv_dir = str(project_dir), str(csv_dir)

    logger = create_logger(project_dir, experiment_name)
    logger.info(f"Starting run {experiment_name} with {model=} "
                f"{tokenization=} {epochs=} {num_hiddens=} device={get_device()}")

    models = {
        'rnn_scratch': RNNModelScratch,
        'rnn_torch': RNNModelPyTorch,
    }

    batch_size = 64

    train_data, vocab_stoi, vocab_itos, vocab_size = dataLoader_for_rq('training', csv_dir,
                                                                       tokenizer_name=tokenizer_name)
    val_data, _, _, _ = dataLoader_for_rq('validation', csv_dir, tokenizer_name=tokenizer_name)
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

    torch.save(net.state_dict(), f"{project_dir}/checkpoints/rq1_{experiment_name}_ep{epochs}.ckpt")

    logger.info(f"------------------")


def train_rnn_from_torch_with_tokenizers():
    """
        train RNN from PyTorch carrying the following tokenizers
        [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel， subword-sentenceLevel]
    """
    run_experiment('rnn_torch-char', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('rnn_torch-word', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('rnn_torch-subword', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword')
    run_experiment('rnn_torch-subword-wordLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='wordLevel')
    run_experiment('rnn_torch-subword-byteLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='byteLevel')
    run_experiment('rnn_torch-subword-charLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='charLevel')
    run_experiment('rnn_torch-subword-sentenceLevel', 'rnn_torch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='sentenceLevel')


def train_rnn_from_scratch_with_tokenizers():
    """
       train RNN from scratch carrying the following tokenizers
       [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel， subword-sentenceLevel]
    """
    run_experiment('rnn_scr-char', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='char')
    run_experiment('rnn_scr-word', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='word')
    run_experiment('rnn_scr-subword', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword')
    run_experiment('rnn_scr-subword-wordLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='wordLevel')
    run_experiment('rnn_scr-subword-byteLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='byteLevel')
    run_experiment('rnn_scr-subword-charLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='charLevel')
    run_experiment('rnn_scr-subword-sentenceLevel', 'rnn_scratch', epochs=100, num_hiddens=128, tokenization='subword',
                   level_type='sentenceLevel')


def train_musk_trump_tokenizers_from_scratch():
    """
        train musk-and trump tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers
    """
    train_tokenizer(training_source_dir='./dataset/train_cleaned.csv', tokenizer_name='BERT',
                    tokenizer_prefix='musk_train')
    train_tokenizer(training_source_dir='./dataset/trump_train.csv', tokenizer_name='BERT',
                    tokenizer_prefix='trump_train')


def train_gpt2_tokenizer_from_scratch():
    """
        train GPT2 tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers
    """
    train_gpt2_tokenizer(training_source_dir='./dataset/trump_train.csv', tokenizer_name='gpt2_trump_tokenizer')
    train_gpt2_tokenizer(training_source_dir='./dataset/train_cleaned.csv', tokenizer_name='gpt2_musk_tokenizer')


if __name__ == '__main__':
    """
       General Training Setup for all training tasks.
       --------------------------------------------------------
       | Epochs | Batch_Size | Step_Size | Optimizer | Metric |
       --------------------------------------------------------
       |  100   |    128     |    0.001  |   Adam    |  ppl.  |
       --------------------------------------------------------
       | ppl. = Perplexity                                    |
       --------------------------------------------------------   
    """

    """
        split the dataset into training-, test- and validation-datasets
    """
    # split_combi_dataset()

    """ 
        train RNN from PyTorch carrying the following tokenizers 
        [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel， subword-sentenceLevel] 
    """
    # train_rnn_from_torch_with_tokenizers()

    """
        train RNN from scratch carrying the following tokenizers 
        [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel， subword-sentenceLevel] 
    """
    # train_rnn_from_scratch_with_tokenizers()

    """
        train musk-and trump tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers
    """
    # train_musk_trump_tokenizers_from_scratch()

    """
        train GPT2 tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers
    """
    # train_gpt2_tokenizer_from_scratch()

    """
        load trained tokenizer
    """
    # tok = Tokenizer.from_file('./my_token/GPT2_trump_train_tokenizer.json')
    # tok.save(path='./my_token/vocab.json')
    # res = tok.encode("Vaccines are just the start. Its also capable in theory of curing almost anything. "
    #                  "fukushima Turns medicine into a software &amp; simulation problem.")
    # print(res.ids)
    # res_ids = res.ids
    # print(tok.decode(res_ids))
    # preprocessing('./dataset/combined_Musks_tweets.csv', './dataset/combined_Musks_tweets_cleaned.csv')
    # csv2txt('./dataset/combined_Musks_tweets_cleaned.csv', './dataset/combined_Musks_tweets_cleaned.txt', 'content')

    """
        load trained GPT2 tokenizer
    """
    # gpt2_trump_tokenizer = Tokenizer.from_file('./my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json')
    # gpt2_musk_tokenizer = Tokenizer.from_file('./my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json')

    """
        calculate the token frequency and compare the frequency difference
    """
    # # [bert_tokenizer]
    # token_counter(tokenizer_dir='./my_token/tokenizers_scratch/BERT_musk_train_tokenizer.json',
    #               csv_dir='./dataset/train_cleaned.csv',
    #               json_dir='./my_token/token_frequency/[musk_train_ds]musk_train_token_freq.json')
    # token_freq_diff(train_json_dir='my_token/token_frequency/[musk_train_ds]musk_train_token_freq.json',
    #                 test_json_dir='my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
    #                 diff_name='[musk_val_ds_diff]musk_musk_train')

    # # [trump/musk tokenizer]
    # token_counter(tokenizer_dir='./my_token/tokenizers_scratch/BERT_musk_train_tokenizer.json',
    #               csv_dir='./dataset/val_cleaned.csv',
    #               json_dir='my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json')
    #
    # token_counter(tokenizer_dir='./my_token/tokenizers_scratch/BERT_trump_train_tokenizer.json',
    #               csv_dir='./dataset/trump_val.csv',
    #               json_dir='my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json')
    #
    # token_freq_diff(train_json_dir='my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json',
    #                 test_json_dir='my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
    #                 diff_name='[trump_musk_ds_diff]trump_musk_train')

    # # [GPT2 tokenizer]

    # # [calculate the freq of trump_train]
    # gpt2_trump_tokenizer = GPT2Tokenizer('./my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
    #                                      './my_token/tokenizers_scratch/gpt2_trump_tokenizer/merges.txt')
    # gpt2_musk_tokenizer = GPT2Tokenizer('./my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
    #                                     './my_token/tokenizers_scratch/gpt2_musk_tokenizer/merges.txt')
    #
    # gpt2_token_counter(tokenizer_dir='./my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
    #                    gpt2_tokenizer=gpt2_trump_tokenizer,
    #                    csv_dir='./dataset/trump_train.csv',
    #                    json_dir='my_token/token_frequency/[trump_train_ds]GPT2_trump_train_token_freq.json')
    #
    # # [calculate the freq of trump_val]
    # gpt2_token_counter(tokenizer_dir='./my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
    #                    gpt2_tokenizer=gpt2_trump_tokenizer,
    #                    csv_dir='./dataset/trump_val.csv',
    #                    json_dir='my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json')
    #
    # # [calculate the freq of musk_train]
    # gpt2_token_counter(tokenizer_dir='./my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
    #                    gpt2_tokenizer=gpt2_musk_tokenizer,
    #                    csv_dir='./dataset/train_cleaned.csv',
    #                    json_dir='my_token/token_frequency/[musk_train_ds]GPT2_musk_train_token_freq.json')
    #
    # # [calculate the freq of musk_val]
    # gpt2_token_counter(tokenizer_dir='./my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
    #                    gpt2_tokenizer=gpt2_musk_tokenizer,
    #                    csv_dir='./dataset/val_cleaned.csv',
    #                    json_dir='my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json')

    # # [calculate the diff between gpt2_trump_val and gpt2_musk_val]
    # token_freq_diff(train_json_dir='my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json',
    #                 test_json_dir='my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json',
    #                 diff_name='[trump_musk_val_ds_diff]GPT2_trump_musk_train')
    #
    # # [calculate the diff between gpt2_trump_val and trump_trump_val]
    # token_freq_diff(train_json_dir='my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json',
    #                 test_json_dir='my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json',
    #                 diff_name='[trump_val_ds_diff]GPT2_trump_trump_train')
    #
    # # [calculate the diff between gpt2_musk_val and musk_musk_val]
    # token_freq_diff(train_json_dir='my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json',
    #                 test_json_dir='my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
    #                 diff_name='[musk_val_ds_diff]GPT2_musk_musk_train')

    """
        load scratch-trained subword tokenizers_scratch and testing the tokenization ability on a given sentence
    """
    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # musk_tokenizer = Tokenizer.from_file('./my_token/BertWordPieceTokenizer_train.json')
    # trump_tokenizer = Tokenizer.from_file('./my_token/BertWordPieceTokenizer_trump_combination.json')
    #
    # trump_tweet = "Wishing you and yours a very Happy and Bountiful Thanksgiving!"
    # print(f'Trump\'s Tweet: {trump_tweet}\n'
    #       f'Bert_Tokenizer: {bert_tokenizer.tokenize(trump_tweet)}\n'
    #       f'Musk_Tokenizer: {musk_tokenizer.encode(trump_tweet).tokens}\n'
    #       f'Trump_Tokenizer: {trump_tokenizer.encode(trump_tweet).tokens}')
    # print()
    # musk_tweet = "Near-orbital space is the fastest way to travel long distance on Earth!"
    # print(f'Musk\'s Tweet: {musk_tweet}\n'
    #       f'Bert_Tokenizer: {bert_tokenizer.tokenize(musk_tweet)}\n'
    #       f'Musk_Tokenizer: {musk_tokenizer.encode(musk_tweet).tokens}\n'
    #       f'Trump_Tokenizer: {trump_tokenizer.encode(musk_tweet).tokens}')

    # print(
    #     bert_tokenizer.tokenize("Vaccines are just the start. Its also capable in theory of curing almost anything. "))
    #
    # brewed_tokenizer = Tokenizer.from_file('./my_token/CharBPETokenizer_Musk_cleaned.json')
    # print(
    #     brewed_tokenizer.encode(
    #         "Vaccines are just the start. Its also capable in theory of curing almost anything. ").tokens)

    # char_tokenizer = lambda str: [char for char in str]
    # word_tokenizer = lambda inp: inp
    # subword_tokenizer = Tokenizer.from_file('./my_token/ByteLevelBPETokenizer_combination.json')
    #
    # text = 'Don\'t you love Transformers? We sure do.'
    # char_tokens = char_tokenizer(text)
    # word_tokens = word_tokenizer(text)
    # subword_tokens = subword_tokenizer.encode(text).tokens
    # sentence_tokens = nltk.sent_tokenize(text)
    #
    # print(
    #     f'char-level: {char_tokens}\n'
    #     f'word-level: {word_tokens}\n'
    #     f'subword-level: {subword_tokens}\n'
    #     f'sentence-level: {sentence_tokens}\n'
    # )

    """
        train Rnn_src_musk_tokenizer
    """
    # run_experiment_for_rq('rnn_src-word-musk-tokenizer', 'rnn_scratch', epochs=100, num_hiddens=128,
    #                       tokenizer_name='musk')
    #

    """
        train Rnn_src_trump_tokenizer
    """
    # run_experiment_for_rq('rnn_src-word-trump-tokenizer', 'rnn_scratch', epochs=100, num_hiddens=128,
    #                       tokenizer_name='trump_combi')

    """
        train GPT2_trump_tokenizer
    """
    # run_experiment_for_rq('rnn_src-gpt2-trump-tokenizer', 'rnn_scratch', epochs=100, num_hiddens=128,
    #                       tokenizer_name='gpt2_trump')

    """
        train GPT2_musk_tokenizer
    """
    # run_experiment_for_rq('rnn_src-gpt2-musk-tokenizer', 'rnn_scratch', epochs=100, num_hiddens=128,
    #                       tokenizer_name='gpt2_musk')
