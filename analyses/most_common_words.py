from pathlib import Path
from utils import get_tokenization_fn
import pandas as pd


def count_words(data_path):
    """
    Count the number of occurrences per word in the given file.

    :param data_path: path to file
    :return: dict of word counts
    """
    path = Path(data_path)

    word_counts = {}
    lines = None
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, delimiter=',')
        lines = list(df['output'])
    elif data_path.endswith(".txt"):
        with open(path) as f:
            lines = f.readlines()

    split = get_tokenization_fn('word')
    for l in lines:
        words = split(l)
        for w in words:
            if w not in word_counts:
                word_counts[w] = 1
            else:
                word_counts[w] += 1

    return word_counts


def top_k_words(data_path, k=100):
    """
    Get a set of the most common k words in the given file.

    :param data_path: file location
    :param k: number of most commmon words
    :return: set of most common words
    """
    word_counts = count_words(data_path)

    return set([k for k, v in sorted(word_counts.items(), key=lambda item: item[1])][-k:])


gt = top_k_words(r'../dataset/combined_Musk_tweets_cleaned.txt')

# fetch the files with the generated tweets.
model_names = ["rnn_scratch", "lstm", "gru", "stacked_lstm"]
tokenizer_names = ["char", "word", "gpt2", "gpt2-trained"]

order = []
k_tops = []
for model in model_names:
    for tokenizer in tokenizer_names:
        k_top = top_k_words(f'../sample_generated_tweets/{model}_{tokenizer}.csv')
        order.append(f"{model}_{tokenizer}")
        k_tops.append(len(gt.intersection(k_top)))

    print(model)
    print(k_tops[-4:])

# do the same calculations for the GPT2 tweets.
file_names = ["gpt2_one_ep.csv", "gpt2_net_scratch.csv", "gpt2_tokenizer_scratch.csv"]
k_tops = []
for file_name in file_names:
    k_top = top_k_words(f"../sample_generated_tweets/{file_name}")
    k_tops.append(len(gt.intersection(k_top)))

print(file_names)
print(k_tops)
