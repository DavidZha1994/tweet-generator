# Installing Python Packages from Jupyter Notebook
# import sys
# !{sys.executable} -m pip3 install torchtext
# !{sys.executable} -m pip3 uninstall -y torchtext
# !python -m pip uninstall torchtext --yes

# Package Settings
import torch
import torch.nn as nn
import preprocessor as pp
import pandas as pd
import timeit
import torchtext  # torchtext==0.4/0.6.0
import logging
import random

PROJECT_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator'
RAW_CSV_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/dataset/realdonaldtrump.csv'
CLEANED_CSV_DIR = '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/dataset/cleaned_tweets.csv'

# Log settings
mylogs = logging.getLogger(__name__)
mylogs.setLevel(logging.INFO)


########################################################################################################################

# Define RNN
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


def pre_processor():
    tweets_csv = pd.read_csv(RAW_CSV_DIR)
    df = pd.DataFrame(tweets_csv)

    cleaned_tweets_list = [pp.clean(content) for content in df['content'] if pp.clean(content) != '']
    cleaned_tweets_dict = {'content': cleaned_tweets_list}
    cleaned_tweets_df = pd.DataFrame(cleaned_tweets_dict)
    cleaned_tweets_df.to_csv(CLEANED_CSV_DIR)

    return cleaned_tweets_df, cleaned_tweets_dict


# homemade dataLoader for csv-files
def brewed_dataLoader(ds_path, ds_format, fields_name):
    text_field = torchtext.data.Field(sequential=True,  # text sequence
                                      tokenize=lambda x: x,  # because are building a character-RNN
                                      include_lengths=True,  # to track the length of sequences, for batching
                                      batch_first=True,
                                      use_vocab=True,  # to turn each character into an integer index
                                      init_token="<BOS>",  # BOS token
                                      eos_token="<EOS>")  # EOS token

    # Set up the fields objects for the TabularDataset
    trump_tweets = torchtext.data.TabularDataset(
        path=ds_path,
        format=ds_format,
        fields=[
            ('', None),  # first column is unnamed
            (fields_name, text_field)
        ])

    text_field.build_vocab(trump_tweets)
    vocab_stoi = text_field.vocab.stoi
    vocab_itos = text_field.vocab.itos
    vocab_size = len(text_field.vocab.itos)

    print("tweets content: ", trump_tweets.examples[6].content)
    print("tweets length: ", len(trump_tweets))
    # print('vocab of text_field: ', text_field)
    print("vocab_stoi: ", vocab_stoi)
    print("vocab_size: ", vocab_size)

    data_iter = torchtext.data.BucketIterator(trump_tweets,
                                              batch_size=1,
                                              sort_key=lambda x: len(x.content),
                                              sort_within_batch=True)
    return trump_tweets, data_iter, vocab_stoi, vocab_itos, vocab_size


def gen_log():
    log_name = '[' + random.randint(0, 100).__str__() + ']' + 'RNN.log'
    print("log file generated: " + log_name)
    file = logging.FileHandler(PROJECT_DIR + "/log/" + log_name)
    file.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
    file.setFormatter(file_format)
    mylogs.addHandler(file)
    # os.getcwd()
    # os.system('tail -f ' + PROJECT_DIR + "/" + log_name)


def train(model, data, batch_size=1, num_epochs=1, lr=0.001, print_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    data_iter = torchtext.data.BucketIterator(data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.content),
                                              sort_within_batch=True)
    ################################## logging #####################################
    gen_log()
    mylogs.info("[Parameter Setting] batch_size={} num_epochs={} step_size={:g} iter={}".format(batch_size,
                                                                                                num_epochs,
                                                                                                lr,
                                                                                                print_every))
    ################################################################################
    it = 0
    for e in range(num_epochs):
        # get training set
        avg_loss = 0
        for (tweet, lengths), label in data_iter:
            target = tweet[:, 1:]
            inp = tweet[:, :-1]
            # cleanup
            optimizer.zero_grad()
            # forward pass
            output, _ = model(inp)
            loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
            # backward pass
            loss.backward()
            optimizer.step()

            avg_loss += loss
            it += 1  # increment iteration count

            if it % print_every == 0:
                ################################## logging #####################################
                mylogs.info("[Epoch {} Iter {}] Loss {:g} Perplexity {:g}".format(e, it + 1,
                                                                                  float(avg_loss / print_every),
                                                                                  torch.exp(
                                                                                      avg_loss / print_every)))
                mylogs.info("[Generated Sequence] {}".format(sample_sequence(model, 140, 0.8)))
                ################################################################################
                # print("[Epoch %d Iter %d] Loss %f" % (e, it + 1, float(avg_loss / print_every)))
                # print("[Generated Sequence] ", sample_sequence(model, 140, 0.8))
                avg_loss = 0


def training_start():
    print('[Starting]')
    start = timeit.default_timer()
    train(model, trump_tweets, batch_size=3, num_epochs=30, lr=0.001, print_every=20000)
    stop = timeit.default_timer()
    print('[Done]')
    print('[Runtime] %.2f ' % float((stop - start) / 60))
    mylogs.info("[Runtime] {:g}".format(float((stop - start) / 60)))

    # Save the model checkpoint
    torch.save(model.state_dict(),
               '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/' + '[' + random.random().__str__() + ']'
               + 'RNN.ckpt')
    print('model saved.done')


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


########################################################################################################################

# Train the Tweet Generator
trump_tweets, data_iter, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader(
    ds_path=CLEANED_CSV_DIR,
    ds_format='csv', fields_name='content')
model = TweetGenerator(vocab_size, hidden_size=64)
training_start()

# Load stored model
# ckpt_model =  torch.save(model.state_dict(),
#                '/Users/yhe/Documents/LocalRepository-Public/tweet-generator/'+'['+datetime.now().__str__()+']'+'rnn.ckpt')
# print(ckpt_model)

# Generate fake tweets
# print(sample_sequence(model, temperature=0.8))
