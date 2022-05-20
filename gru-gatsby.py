import math
import time

import torch
from torch import nn
from torch.nn import functional as F
import utils
from utils import Accumulator

batch_size, num_steps = 32, 35
train_iter, vocab = utils.load_data_gatsby(batch_size, num_steps)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RNNModelScratch(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(RNNModelScratch, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        # reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit#Fully_gated_unit
        self.i2z = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.i2r = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.i2h = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.h2o = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        H, = state
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            z = torch.sigmoid(self.i2z(torch.cat((X_step, H), 1)))
            r = torch.sigmoid(self.i2r(torch.cat((X_step, H), 1)))
            h = torch.tanh(self.i2h(torch.cat((X_step, r * H), 1)))
            H = z * h + (1 - z) * H
            Y = self.h2o(H)

            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, num_hiddens), device=device),


def predict_ch8(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, loss, updater, device):
    """Train a net within one epoch"""
    start_time = time.time()
    metric = Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        # create new hidden state at start of each epoch
        state = net.begin_state(batch_size=X.shape[0], device=device)

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / (time.time() - start_time)


def train(net, train_iter, vocab, lr, num_epochs, device):
    """Train a model"""
    loss = nn.CrossEntropyLoss()

    # Initialize
    optimizer = torch.optim.Adam(net.parameters(), lr)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(predict('she wanted'))

            print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('she wanted'))
    print(predict('she'))


num_hiddens = 256
net = RNNModelScratch(len(vocab), num_hiddens, get_device())
num_epochs, lr = 500, 0.001
train(net, train_iter, vocab, lr, num_epochs, get_device())
