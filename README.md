# tweet-generator

### setup
To run the project make sure to install python 3.9 (this is easiest with anaconda/miniconda).<br>
The requirements are found in the 'requirements.txt' file and can be installed into the environment by running 'python -m pip install -r requirements.txt
'<br>

### train models
The training of the nets is split into two files, one for the RNNs and one for the transformers. These are 'rnn_training.py' and 'gpt2_training.py'.
<br>
Executing the 'rnn_training.py' file will train all RNNs combined with all tokenizers, i.e. 4x4 models.
<br>
The trained RNNs can afterwards be found in the checkpoints directory <br>
They are named based on the model (lstm->LSTM, gru->GRU, rnn_scr->vanilla RNN, lstm_stacked->stacked version of the LSTM) and tokenizer (char->character level, word->word level, gpttoken->pretrained subword, gptword->custom subword).
<br>
<br>
To train the gpt2 use the 'gpt2_training.py' file. By setting the booleans 'TRAIN_NN_FROM_SCRATCH' and 'TRAIN_TOKENIZER_FROM_SCRATCH'
accordingly it can be decided how the net should be trained.
<br>
The results will be saved in the models directory.

### generate tweets
As for the training there are two files to generate tweets depending on the architecture type.
<br>
To run the RNNs use 'rnn_predict.py'.
<br>
Here the
1. tokenizer type ('char'->character level, 'word'->word level, 'gpt2'->pre trained gpt2 tokenizer, 'gpt2-trained')
<br>
2. selected model (lstm, rnn_scratch, gru, stacked_lstm)
3. model path (path to the checkpoint where the model is stored)
<br>

must be set accordingly to the model for which the generation should be done.
(This might be a bit tedious.)
<br>

Generating tweets with the GPT2 is a bit easier. In the 'gpt2_predict.py' file set the 'selection' to one of these: 'gpt2_one_ep' (fine tuned for one epoch), 'gpt2_fine_tuned' (fine tuned dor 100 epochs), 'gpt2_net_scratch' (net trained from scratch), 'gpt2_tokenizer_scratch' (net&tokenizer trained from scratch)
<br>

The generated tweets can then be found in the directory 'sample_generated tweets'.

### analyses
The files for the analyses can be found in the 'analyses' directory. When the tweets are generated there are no further adjustments needed, and they can just be executed.