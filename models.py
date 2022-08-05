import torch  # pytorch == 1.11.0
import torch.nn as nn
from torch.nn import functional as F


class TweetGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', n_layers=1):
        super(TweetGenerator, self).__init__()

        # identiy matrix for generating one-hot vectors
        self.ident = torch.eye(input_size, device=device)
        self.input_size = input_size

        # recurrent neural network
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            device=device
        )
        # FC layer as decoder to the output
        self.fc = nn.Linear(hidden_size, input_size, device=device)

    def forward(self, x, h_state=None):
        x = self.ident[x]  # generate one-hot vectors of input
        output, h_state = self.rnn(x, h_state)  # get the next output and hidden state
        output = self.fc(output)  # predict distribution over next tokens
        output = output.reshape(-1, self.input_size)  # use the same shape as the other nets
        return output, h_state


class RNNModelScratch(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device='cpu'):
        super(RNNModelScratch, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.device = device

        self.i2h = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.h2o = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, state=None):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        if state is None:
            state = self.begin_state(X.shape[1], self.device),

        H, = state
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            H = torch.tanh(self.i2h(torch.cat((X_step, H), 1)))
            Y = self.h2o(H)
            outputs.append(Y)

        return torch.cat(outputs, dim=0), (H,)

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, self.num_hiddens), device=device)


class LstmCell(nn.Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super(LstmCell, self).__init__()

        # output size determines the number of hidden units (hidden_size = output_size)
        # input_size = output_size = vocab_size
        # self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.input_size, self.output_size = input_size, output_size
        num_hiddens = output_size

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


class LSTM(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(LSTM, self).__init__()

        input_size = output_size = vocab_size
        self.device = device
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        self.cell1 = LstmCell(input_size, num_hiddens, device)
        self.cell2 = LstmCell(num_hiddens, num_hiddens, device)
        self.h2out = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, states=None):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        if states is None:
            state1 = self.begin_state(X.shape[1], self.device)
            state2 = self.begin_state(X.shape[1], self.device)
        else:
            state1, state2 = states

        H_1, C_1 = state1
        H_2, C_2 = state2
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            (H_1, C_1) = self.cell1(X_step, (H_1, C_1))
            (H_2, C_2) = self.cell2(H_1, (H_2, C_2))
            Y = self.h2out(H_2)
            outputs.append(Y)
        return torch.cat(outputs, dim=0), ((H_1, C_1), (H_2, C_2))

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, self.num_hiddens), device=device), torch.zeros((batch_size, self.num_hiddens),
                                                                                       device=device)


class StackedLstm(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(StackedLstm, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.device = device

        self.cell1 = LstmCell(input_size, num_hiddens, device=device)
        self.cell2 = LstmCell(num_hiddens, num_hiddens, device=device)
        self.h2out = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, states=None):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        if states is None:
            state1 = self.begin_state(X.shape[1], self.device)
            state2 = self.begin_state(X.shape[1], self.device)
        else:
            state1, state2 = states

        H_1, C_1 = state1
        H_2, C_2 = state2
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            (H_1, C_1) = self.cell1(X_step, (H_1, C_1))
            (H_2, C_2) = self.cell2(H_1, (H_2, C_2))
            Y = self.h2out(H_2)
            outputs.append(Y)
        return torch.cat(outputs, dim=0), ((H_1, C_1), (H_2, C_2))

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, self.num_hiddens), device=device), torch.zeros((batch_size, self.num_hiddens),
                                                                                       device=device)

    def begin_rand_state(self, batch_size, device):
        return torch.rand((batch_size, self.num_hiddens), device=device), torch.rand((batch_size, self.num_hiddens),
                                                                                     device=device)

class StackedLstm3(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(StackedLstm3, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.device = device

        self.cell1 = LstmCell(input_size, num_hiddens, device=device)
        self.cell2 = LstmCell(num_hiddens, num_hiddens, device=device)
        self.cell3 = LstmCell(num_hiddens, num_hiddens, device=device)
        self.h2out = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, states=None):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        if states is None:
            state1 = self.begin_state(X.shape[1], self.device)
            state2 = self.begin_state(X.shape[1], self.device)
            state3 = self.begin_state(X.shape[1], self.device)
        else:
            state1, state2, state3 = states

        H_1, C_1 = state1
        H_2, C_2 = state2
        H_3, C_3 = state3
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            (H_1, C_1) = self.cell1(X_step, (H_1, C_1))
            (H_2, C_2) = self.cell2(H_1, (H_2, C_2))
            (H_3, C_3) = self.cell3(H_2, (H_3, C_3))
            Y = self.h2out(H_3)
            outputs.append(Y)
        return torch.cat(outputs, dim=0), ((H_1, C_1), (H_2, C_2), (H_3, C_3))

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, self.num_hiddens), device=device), \
               torch.zeros((batch_size, self.num_hiddens),device=device),

    def begin_rand_state(self, batch_size, device):
        return torch.rand((batch_size, self.num_hiddens), device=device), \
               torch.rand((batch_size, self.num_hiddens),device=device),


class GRU(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device):
        super(GRU, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.device = device

        # reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit#Fully_gated_unit
        self.i2z = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.i2r = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.i2h = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.h2o = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, state=None):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        if state is None:
            state = self.begin_state(X.shape[1], self.device)

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
        return torch.zeros((batch_size, self.num_hiddens), device=device),
