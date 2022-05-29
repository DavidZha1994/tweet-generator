import math
import time

import torch
from torch import nn
from torch.nn import functional as F
import utils
from utils import Accumulator
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size, num_steps = 128, 35
train_iter, vocab = utils.load_data_gatsby(batch_size, num_steps, start_token=0, max_tokens=.9, relative_size=True)
val_iter, _ = utils.load_data_gatsby(batch_size, num_steps, start_token=0.9, max_tokens=1, relative_size=True)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LstmCell(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(LstmCell, self).__init__()

        # output size determines the number of hidden units (hidden_size = output_size)
        # input_size = output_size = vocab_size
        # self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.input_size, self.output_size = input_size, output_size

        # reference: https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        self.in2f = nn.Linear(input_size + num_hiddens, output_size, device=device)
        self.in2i = nn.Linear(input_size + num_hiddens, output_size, device=device)
        self.in2o = nn.Linear(input_size + num_hiddens, output_size, device=device)
        self.in2c_tilde = nn.Linear(input_size + output_size, output_size, device=device)

    def forward(self, X, state):
        H, C = state

        f = torch.sigmoid(self.in2f(torch.concat((X, H), 1)))
        i = torch.sigmoid(self.in2i(torch.concat((X, H), 1)))
        o = torch.sigmoid(self.in2o(torch.concat((X, H), 1)))

        c_tilde = torch.tanh(self.in2c_tilde(torch.concat((X, H), 1)))
        C = f * C + i * c_tilde
        H = o * torch.tanh(C)

        return (H, C)


class RNNModelScratch(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(RNNModelScratch, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        self.cell1 = LstmCell(input_size, num_hiddens, device)
        self.cell2 = LstmCell(num_hiddens, num_hiddens, device)
        self.h2out = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, state1, state2):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        H_1, C_1 = state1
        H_2, C_2 = state2
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            (H_1, C_1) = self.cell1(X_step, (H_1, C_1))
            (H_2, C_2) = self.cell2(H_1, (H_2, C_2))
            Y = self.h2out(H_2)
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H_1, C_1), (H_2, C_2)

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, num_hiddens), device=device), torch.zeros((batch_size, num_hiddens),
                                                                                  device=device)


def predict(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`."""
    state_1 = net.begin_state(batch_size=1, device=device)
    state_2 = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state_1, state_2 = net(get_input(), state_1, state_2)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state_1, state_2 = net(get_input(), state_1, state_2)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


@torch.no_grad()
def evaluate(net, val_iter, loss, device):
    metric = Accumulator(2)

    for X, Y in val_iter:
        state_1 = net.begin_state(batch_size=X.shape[0], device=device)
        state_2 = net.begin_state(batch_size=X.shape[0], device=device)

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state_1, state_2 = net(X, state_1, state_2)
        l = loss(y_hat, y.long()).mean()

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1])


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

    for X, Y in tqdm(train_iter, desc='training'):
        # create new hidden state at start of each batch
        state_1 = net.begin_state(batch_size=X.shape[0], device=device)
        state_2 = net.begin_state(batch_size=X.shape[0], device=device)

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state_1, state_2 = net(X, state_1, state_2)
        l = loss(y_hat, y.long()).mean()

        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

        # this way of computing the perplexity works only because the cross-entropy loss is used!
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / (time.time() - start_time)


def train(net, train_iter, vocab, lr, num_epochs, device):
    """Train a model"""
    loss = nn.CrossEntropyLoss()
    train_results = []
    val_results = []

    # Initialize
    optimizer = torch.optim.Adam(net.parameters(), lr)

    predict_seq = lambda prefix: predict(prefix, 100, net, vocab, device)
    # Train and predict
    for epoch in tqdm(range(num_epochs), desc='epoch'):
        train_ppl, speed = train_epoch(
            net, train_iter, loss, optimizer, device)
        train_results.append(train_ppl)

        val_ppl = evaluate(net, val_iter, loss, device)
        val_results.append(val_ppl)
        print(f'epoch {epoch} - val-ppl: {val_ppl:.1f}, train-ppl: {train_ppl:.1f}')

        if (epoch + 1) % 10 == 0:
            print(predict_seq('she wanted'))
            print(predict_seq('i am'))
            # print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
            # plt results
            plt.plot(train_results, label='train')
            plt.xlabel('epoch')
            plt.ylabel('perplexity')
            plt.plot(val_results, label='val')
            plt.legend()
            plt.show()

    # inference after training
    print(predict_seq('she wanted'))
    print(predict_seq('she'))

    # plt results
    plt.plot(train_results, label='train')
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.plot(val_results, label='val')
    plt.legend()
    plt.show()


num_hiddens = 256
net = RNNModelScratch(len(vocab), num_hiddens, get_device())

num_epochs, lr = 500, 0.001
train(net, train_iter, vocab, lr, num_epochs, get_device())