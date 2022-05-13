import random

import torch


def load_data_gatsby() -> tuple:
    with open('./dataset/gatsby.txt', 'r', encoding="utf8") as file:
        data = file.read().replace('\n', '')

    data_processed = ""
    data.replace('-', '')

    vocab = set()

    last_char = ''
    for idx, char in enumerate(data):
        vocab.add(char)
        if not (char == '-' == last_char or char == ' ' == last_char):
            data_processed += char
        last_char = char

    return data_processed, list(vocab)


def input_tensor(seq: str, vocab: list):
    tensor = torch.zeros(len(seq), 1, len(vocab))
    for char_idx in range(len(seq)):
        char = seq[char_idx]
        tensor[char_idx][0][vocab.index(char)] = 1

    return tensor


def target_tensor(seq: str, vocab: list):
    indices = [vocab.index(char) for char in seq]
    return torch.LongTensor(indices)


# return x, y data point (a sequence of defined length)
def dg_gatsby(seq_len=20):
    corpus, vocab = load_data_gatsby()
    corpus_len = len(corpus)

    while True:
        start_idx = random.randint(0, corpus_len - seq_len - 1)
        yield input_tensor(corpus[start_idx:start_idx + seq_len], vocab), target_tensor(
            corpus[start_idx + 1:start_idx + seq_len + 1], vocab)


if __name__ == '__main__':
    # just some testing
    corpus, vocab = load_data_gatsby()
    input = input_tensor("bac", ["a", "b", "c", "d"])
    target = target_tensor("bac", ["a", "b", "c", "d"])
    print(input)
    print(target.unsqueeze(-1).shape)
