import torch
from models import StackedLstm, LSTM, GRU, RNNModelScratch
from utils import brewed_dataLoader, get_tokenization_fn
import pandas as pd

# Make sure to select the tokenizer, model and path accordingly
TOKENIZER_TYPE = 'gpt2-trained'             # 'gpt2' for huggingface, 'gpt2-trained' for scratch
SELECTED_MODEL = 'stacked_lstm'
MODEL_PATH = 'checkpoints/lstm_stacked-gpt-word_ep100.ckpt'  # 'gpttoken' for huggingface,  'gpt-word' for scratch

csv_dir = r'dataset'
_, vocab_stoi, vocab_itos, vocab_size = brewed_dataLoader('training', csv_dir, tokenization=TOKENIZER_TYPE)
vocab = vocab_itos, vocab_stoi, vocab_size

models = {
    'rnn_scratch': RNNModelScratch,
    'lstm': LSTM,
    'stacked_lstm': StackedLstm,
    'gru': GRU,
}
model = models[SELECTED_MODEL](vocab_size, 128, 'cuda:0')
model.load_state_dict(torch.load(MODEL_PATH))


def predict(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`."""
    vocab_itos, vocab_stoi, _ = vocab
    tokenize_fn = get_tokenization_fn(TOKENIZER_TYPE)
    prefix = tokenize_fn(prefix)

    state = None
    outputs = [vocab_stoi[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab_stoi[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        y_sorted = torch.argsort(y, descending=True)

        if int(y_sorted[0][0]) != vocab_stoi['<EOS>']:
            outputs.append(int(torch.argsort(y, descending=True)[0][0]))
        else:
            outputs.append(int(y_sorted[0][1]))

    return format_output(outputs, TOKENIZER_TYPE, vocab_itos)


def format_output(output, tokenization, vocab_itos):
    if tokenization == 'char':
        # concat all characters (spaces already contained in outputs)
        return "".join(vocab_itos[i] for i in output)
    if tokenization == 'word':
        # concat all words (need to add spaces between all words)
        return " ".join(vocab_itos[i] for i in output)
    if tokenization == 'gpt2-trained' or tokenization == 'gpt2':
        # at the beginning of each new word is a 'Ġ'-character, replace them with a space
        formatted = "".join(vocab_itos[i] for i in output)
        formatted = formatted.replace('\u0120', ' ')
        return formatted
    if tokenization == 'subword':
        # subwords are concatenated if they begin with ##, so we can subtract these from the sentence length
        formatted = " ".join([vocab_itos[i] for i in output])
        formatted = formatted.replace(" ##", '')
        return formatted


inputs = ["This", "I", "Tesla", "Entertainment", "Just", "We", "A", "Haha", "There", "Good", "Thanks", "Bitcoin",
          "Ascent", "Yeah", "Major", "Tomorrow", "Landing", "Indeed", "Maybe", "Possible", "Burst", "Autopilot",
          "China", "Also", "London", "Roughly"]
outputs = []
for inp in inputs:
    output = predict(inp, 50, model, vocab, 'cuda:0')
    outputs.append(output)
    print(output)

data = [[prompt, output] for prompt, output in zip(inputs, outputs)]
df = pd.DataFrame(data=data, columns=['prompt', 'output'])

df.to_csv(f"./sample_generated_tweets/{SELECTED_MODEL}_{TOKENIZER_TYPE}.csv", mode="w+")
