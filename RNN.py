# Package Settings
import time

import torch  # pytorch == 1.11.0
import torch.nn as nn
import preprocessor as pp  # tweet-preprocessor == 0.6.0
import pandas as pd  # pandas == 1.4.2
import timeit
import torchtext  # torchtext==0.4/0.6.0
import logging
import random
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForMaskedLM

# Directories Settings
PROJECT_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator'
RAW_CSV_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/dataset/elons_val.csv'
CLEANED_CSV_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/dataset/'

# Hyperparameter Settings
batch_size = 21  # (training) 16779 = 3 * 7 * 17 * 47 , (validation) 179
num_epochs = 30  # each epoch = BatchSize * Iteration
lr = 0.001
iterations = 799 + 1
num_hiddens = 64  # 64

# Log settings
mylogs = logging.getLogger(__name__)
mylogs.setLevel(logging.INFO)


########################################################################################################################
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Default RNN
class TweetGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(TweetGenerator, self).__init__()

        # identiy matrix for generating one-hot vectors
        self.ident = torch.eye(input_size)

        # recurrent neural network
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True
        )
        # FC layer as decoder to the output
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_state=None):
        x = self.ident[x]  # generate one-hot vectors of input
        output, h_state = self.rnn(x, h_state)  # get the next output and hidden state
        output = self.fc(output)  # predict distribution over next tokens
        return output, h_state


class RNNModelScratch(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device=get_device()):
        super(RNNModelScratch, self).__init__()

        input_size = output_size = vocab_size
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        self.i2h = nn.Linear(input_size + num_hiddens, num_hiddens, device=device)
        self.h2o = nn.Linear(num_hiddens, output_size, device=device)

    def forward(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # Shape of `X`: (`sequence_size`,`batch_size`, `vocab_size`)

        H, = state
        outputs = []
        # Shape of `X_step`: (`batch_size`, `vocab_size`)
        for X_step in X:
            H = torch.tanh(self.i2h(torch.cat((X_step, H), 1)))
            Y = self.h2o(H)
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    def begin_state(self, batch_size, device):
        return torch.zeros((batch_size, num_hiddens), device=device),


def pre_processor():
    tweets_csv = pd.read_csv(RAW_CSV_DIR)
    df = pd.DataFrame(tweets_csv)
    # 'content' or 'tweet'
    cleaned_tweets_list = [pp.clean(content) for content in df['tweet'] if pp.clean(content) != '']
    cleaned_tweets_dict = {'content': cleaned_tweets_list}
    cleaned_tweets_df = pd.DataFrame(cleaned_tweets_dict)
    cleaned_tweets_df.to_csv(CLEANED_CSV_DIR + 'cleaned_elons_tweet.csv')

    return cleaned_tweets_df, cleaned_tweets_dict


# homemade dataLoader for csv-files
def brewed_dataLoader(which_data):  # which_ds could be 'training', 'validation'

    # Subword-based tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Character-based tokenization
    # tokenize = lambda x:x
    # Word-based tokenization
    # tokenize = lambda x: x.split()

    # it is for character/word-based tokenization
    text_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=tokenizer.tokenize,  # because are building a character/subword/word-RNN
                                      include_lengths=True,  # to track the length of sequences, for batching
                                      batch_first=True,
                                      use_vocab=True,  # to turn each character/word/subword into an integer index
                                      init_token="<BOS>",  # BOS token
                                      eos_token="<EOS>",  # EOS token
                                      unk_token=None)

    train_data, val_data = torchtext.data.TabularDataset.splits(
        path=CLEANED_CSV_DIR,
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


def gen_log():
    log_name = '[' + random.randint(0, 100).__str__() + ']' + 'RNN.log'
    print("log file generated: " + log_name)
    file = logging.FileHandler(PROJECT_DIR + "/log/" + log_name)
    file.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
    file.setFormatter(file_format)
    mylogs.addHandler(file)


def train(model, training_data, val_data, vocab_size, batch_size=1, num_epochs=1, lr=0.001,
          print_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    data_iter = torchtext.data.BucketIterator(training_data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.content),
                                              sort_within_batch=True)
    val_iter = torchtext.data.BucketIterator(val_data,
                                             batch_size=32,
                                             sort_key=lambda x: len(x.content),
                                             sort_within_batch=True)

    ################################## logging #####################################
    gen_log()
    mylogs.info("[Hyperparameter Settings] batch_size={} num_epochs={} step_size={:g} iter={}".format(batch_size,
                                                                                                      num_epochs,
                                                                                                      lr,
                                                                                                      print_every))
    ################################################################################
    it = 0
    for e in tqdm(range(num_epochs), desc='epoch'):
        avg_loss = 0.0
        avg_val_loss = 0.0

        for (tweet, lengths), label in tqdm(data_iter, desc='training'):
            target = tweet[:, 1:]
            inp = tweet[:, :-1]

            # forward pass
            # state = model.begin_state(batch_size=batch_size, device=get_device())
            # output, state = model(inp, state)
            output, _ = model(inp)
            loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss
            it += 1  # increment iteration count

            if it % print_every == 0:
                # val_state = model.begin_state(batch_size=batch_size, device=get_device())
                for (val_tweet, _), _ in tqdm(val_iter, desc='validation'):
                    val_target = val_tweet[:, 1:]
                    val_inp = val_tweet[:, :-1]

                    # forward pass
                    # val_output, _ = model(val_inp, val_state)
                    val_output, _ = model(val_inp)
                    val_loss = criterion(val_output.reshape(-1, vocab_size), val_target.reshape(-1))
                    avg_val_loss += val_loss

                ################################## logging #####################################
                mylogs.info("[Epoch {}] Loss {:f} Val_Loss {:f} Perplexity {:g}".format(e,
                                                                                        float(
                                                                                            avg_loss / len(data_iter)),
                                                                                        float(
                                                                                            avg_val_loss / len(
                                                                                                val_iter)),
                                                                                        torch.exp(
                                                                                            avg_loss / len(data_iter))
                                                                                        )
                            )
                mylogs.info("[Generated Sequence] {}".format(sample_sequence(model, 140, 0.8)))
                ################################################################################
                avg_loss, avg_val_loss = 0, 0  # reset two loss values to zero


def training_start(model, training_iter, val_iter, vocab_size, batch=1, epochs=1,
                   lr=0.001, iterations=100):
    print('[Starting]')
    start = timeit.default_timer()
    train(model, training_iter, val_iter, vocab_size, batch, epochs, lr, iterations)
    stop = timeit.default_timer()
    print('[Done]')
    print('[Runtime] %.2f ' % float((stop - start) / 60))
    mylogs.info("[Runtime] {:g}".format(float((stop - start) / 60)))

    # Save the model checkpoint
    torch.save(model.state_dict(),
               PROJECT_DIR + '/' + '[' + random.random().__str__() + ']'
               + 'RNN.ckpt')
    print('model saved.done')


# A sequence generator for string/character prediction
def sample_sequence(model, max_len=100, temperature=0.8):
    generated_sequence = ""
    # one-to-many
    inp = torch.Tensor([vocab_stoi["<BOS>"]]).long()
    hidden = None
    for p in range(max_len):
        output, hidden = model(inp.unsqueeze(0), hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted character to string and use as next input
        predicted_char = vocab_itos[top_i]

        if predicted_char == "<EOS>":
            break
        generated_sequence += predicted_char
        inp = torch.Tensor([top_i]).long()
    return generated_sequence


# A sequence generator for word prediction
def sample_word_sequence(model, max_len=100, temperature=0.8):
    # TODO: define a seed input
    generated_sequence = ""
    # one-to-many
    inp = torch.Tensor([vocab_stoi["<BOS>"]]).long()
    for p in range(max_len):
        output, hidden = model(inp.unsqueeze(0))
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted word to string and use as next input
        predicted_word = vocab_itos[top_i]

        if predicted_word == "<EOS>":
            break
        generated_sequence += ' ' + predicted_word
        inp = torch.Tensor([top_i]).long()
    return generated_sequence


########################################################################################################################

# Train the Tweet Generator
training_data, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('training')
val_data, _, _, _ = brewed_dataLoader('validation')
# testing_data, testing_iter, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('testing')

model = TweetGenerator(vocab_size, hidden_size=num_hiddens)
# model = RNNModelScratch(vocab_size, num_hiddens, get_device())
training_start(model, training_data, val_data, vocab_size, batch_size, num_epochs, lr,
               iterations)

# Load stored model
# ckpt_model =  torch.save(model.state_dict(),
#                PROJECT_DIR + '/' + '[' + datetime.now().__str__()+'] '+' rnn.ckpt')
# print(ckpt_model)

# Generate fake tweets
# print(sample_sequence(model, temperature=0.8))
