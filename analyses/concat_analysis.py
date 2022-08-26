import numpy as np
from utils import get_tokenization_fn
import torch
from torch import nn
import time

# concatenation vs summation
BATCH_SIZE = 64
INPUT_SIZE = 1000
NUM_HIDDEN = 100
device = 'cuda:0'

x = torch.randn((BATCH_SIZE, INPUT_SIZE), device=device)
h = torch.randn((BATCH_SIZE, NUM_HIDDEN), device=device)


def timeit(func):
    def decorated(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        stop = time.time()
        print(f"{func.__name__}: {stop - start:.2f}s")

    return decorated


@timeit
def concat(x, h):
    linear = nn.Linear(INPUT_SIZE + NUM_HIDDEN, NUM_HIDDEN, device=device)

    for _ in range(500000):
        linear(torch.concat((x, h), 1))


@timeit
def add(x, h):
    linear_w = nn.Linear(INPUT_SIZE, NUM_HIDDEN, device=device)
    linear_u = nn.Linear(NUM_HIDDEN, NUM_HIDDEN, device=device)

    for _ in range(500000):
        linear_w(x) + linear_u(h)


# run time:
concat(x, h)
add(x, h)
