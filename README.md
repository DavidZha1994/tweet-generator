# Tokenization Analysis
## Environmental Settings
Before running the project, please check your package list to make sure the packages are of the same version, as shown below. Alternatively, you can install the listed packages by running command ```python -m pip install -r requirements.txt ```. 
```bash
python 3.9
cleantext          1.1.4
datasets           2.3.2
Levenshtein        0.20.2
nltk               3.7
numpy              1.23.0
pandas             1.4.3
preprocessor       1.1.3
regex              2022.6.2
tokenizers         0.12.1
torch              1.12.0
torchaudio         0.14.0.dev20220603
torchtext          0.4.0
torchvision        0.14.0.dev20220702
tqdm               4.64.0
transformers       4.20.1
```

To train the models related to tokenization analysis, please go into the ```tokenization_analysis.py``` and run :
```python
# train RNN from PyTorch carrying the following tokenizers 
# [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel, subword-sentenceLevel] 

train_rnn_from_torch_with_tokenizers()

# train RNN from scratch carrying the following tokenizers 
# [char, word, subword, subword-wordLevel， subword-byteLevel， subword-charLevel， subword-sentenceLevel] 

train_rnn_from_scratch_with_tokenizers()

# train musk-and trump tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers

train_musk_trump_tokenizers_from_scratch()

# train GPT2 tokenizer from scratch by initializing BertWordPieceTokenizer from Transformers

train_gpt2_tokenizer_from_scratch()
```
To view the tokenization analysis you can check the generated files in the directory ```./tokenization_analysis/RQ1_Result``` and ```./tokenization_analysis/RQ2_Result```, or you can run the program yourself with running the following:
```python
# load trained tokenizer

tok = Tokenizer.from_file('./tokenization_analysis/my_token/GPT2_trump_train_tokenizer.json')

# load trained GPT2 tokenizer

gpt2_trump_tokenizer = Tokenizer.from_file('./tokenization_analysis/my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json')
gpt2_musk_tokenizer = Tokenizer.from_file('./tokenization_analysis/my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json')

# calculate the token frequency and compare the frequency difference

# [bert_tokenizer]
token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/BERT_musk_train_tokenizer.json',
              csv_dir='./dataset/train_cleaned.csv',
              json_dir='./tokenization_analysis/my_token/token_frequency/[musk_train_ds]musk_train_token_freq.json')
token_freq_diff(train_json_dir='./tokenization_analysis/my_token/token_frequency/[musk_train_ds]musk_train_token_freq.json',
                test_json_dir='./tokenization_analysis/my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
                diff_name='[musk_val_ds_diff]musk_musk_train')

# [trump/musk tokenizer]
token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/BERT_musk_train_tokenizer.json',
              csv_dir='./dataset/val_cleaned.csv',
              json_dir='/tokenization_analysis/my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json')

token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/BERT_trump_train_tokenizer.json',
              csv_dir='./dataset/trump_val.csv',
              json_dir='/tokenization_analysis/my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json')

token_freq_diff(train_json_dir='/tokenization_analysis/my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json',
                test_json_dir='/tokenization_analysis/my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
                diff_name='[trump_musk_ds_diff]trump_musk_train')

# [GPT2 tokenizer]

# [calculate the freq of trump_train]
gpt2_trump_tokenizer = GPT2Tokenizer('./tokenization_analysis/my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
                                     './tokenization_analysis/my_token/tokenizers_scratch/gpt2_trump_tokenizer/merges.txt')
gpt2_musk_tokenizer = GPT2Tokenizer('./tokenization_analysis/my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
                                    './tokenization_analysis/my_token/tokenizers_scratch/gpt2_musk_tokenizer/merges.txt')

gpt2_token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
                   gpt2_tokenizer=gpt2_trump_tokenizer,
                   csv_dir='./dataset/trump_train.csv',
                   json_dir='./tokenization_analysis/my_token/token_frequency/[trump_train_ds]GPT2_trump_train_token_freq.json')

# [calculate the freq of trump_val]
gpt2_token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/gpt2_trump_tokenizer/vocab.json',
                   gpt2_tokenizer=gpt2_trump_tokenizer,
                   csv_dir='./dataset/trump_val.csv',
                   json_dir='./tokenization_analysis/my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json')

# [calculate the freq of musk_train]
gpt2_token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
                   gpt2_tokenizer=gpt2_musk_tokenizer,
                   csv_dir='./dataset/train_cleaned.csv',
                   json_dir='./tokenization_analysis/my_token/token_frequency/[musk_train_ds]GPT2_musk_train_token_freq.json')

# [calculate the freq of musk_val]
gpt2_token_counter(tokenizer_dir='./tokenization_analysis/my_token/tokenizers_scratch/gpt2_musk_tokenizer/vocab.json',
                   gpt2_tokenizer=gpt2_musk_tokenizer,
                   csv_dir='./dataset/val_cleaned.csv',
                   json_dir='./tokenization_analysis/my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json')

# [calculate the diff between gpt2_trump_val and gpt2_musk_val]
token_freq_diff(train_json_dir='./tokenization_analysis/my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json',
                test_json_dir='./tokenization_analysis/my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json',
                diff_name='[trump_musk_val_ds_diff]GPT2_trump_musk_train')

# [calculate the diff between gpt2_trump_val and trump_trump_val]
token_freq_diff(train_json_dir='./tokenization_analysis/my_token/token_frequency/[trump_val_ds]GPT2_trump_train_token_freq.json',
                test_json_dir='./tokenization_analysis/my_token/token_frequency/[trump_val_ds]trump_train_token_freq.json',
                diff_name='[trump_val_ds_diff]GPT2_trump_trump_train')

# [calculate the diff between gpt2_musk_val and musk_musk_val]
token_freq_diff(train_json_dir='./tokenization_analysis/my_token/token_frequency/[musk_val_ds]GPT2_musk_train_token_freq.json',
                test_json_dir='./tokenization_analysis/my_token/token_frequency/[musk_val_ds]musk_train_token_freq.json',
                diff_name='[musk_val_ds_diff]GPT2_musk_musk_train')
```
