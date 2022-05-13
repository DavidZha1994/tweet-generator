import torch
import torch.nn as nn

from utils import load_data_gatsby, dg_gatsby, input_tensor
import matplotlib.pyplot as plt


class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o2 = nn.Linear(output_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, input, hidden):
        in_combined = torch.concat((input, hidden), dim=1)
        hidden = self.i2h(in_combined)
        output = self.i2o(in_combined)
        out_combined = torch.concat((output, hidden), dim=1)
        output = self.o2o2(out_combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# train on a single batch (i.e. 1 sequence since batchsize=1)
def train(input_tensor, target_tensor, rnn: GRNN):
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    loss = 0  # final loss

    rnn.zero_grad()

    for i in range(len(target_tensor)):
        prediction, hidden = rnn(input_tensor[i], hidden)
        l = criterion(prediction, target_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return prediction, loss.item() / input_tensor.size(0)


def main():
    _, vocab = load_data_gatsby()
    vocab_size = len(vocab)
    model = GRNN(vocab_size, 128, vocab_size)
    dg = dg_gatsby()  # data generator

    avg_loss_history = []
    avg_loss = 0

    for i in range(1, 100000):
        x, y = next(dg)

        y_hat, loss = train(x, y, model)
        avg_loss += loss

        if i % 500 == 0:
            print(f"iteration: {i} - loss: {avg_loss / 500}")
            avg_loss_history.append(avg_loss / 500)
            avg_loss = 0

    plt.plot(avg_loss_history)
    plt.show()

    # inference:
    hidden = model.init_hidden()
    startSeq = input_tensor("He ", vocab)
    finalWord = "He "

    with torch.no_grad():
        # get model started:
        for i in range(len(startSeq)):
            pred, hidden = model(startSeq[i], hidden)
            idx, _ = pred.squeeze(0).topk(1)
            pred_char = vocab[int(idx[0].item())]
            print(pred_char)

        # predict new characters
        for i in range(20):
            in_tensor = input_tensor(finalWord[-1], vocab)[0]
            pred, hidden = model(in_tensor, hidden)
            idx, _ = pred.squeeze(0).topk(1)
            pred_char = vocab[int(idx[0].item())]
            finalWord += pred_char
            print(finalWord)



if __name__ == '__main__':
    criterion = nn.NLLLoss()
    lr = 0.0005

    # TODO make use of gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
