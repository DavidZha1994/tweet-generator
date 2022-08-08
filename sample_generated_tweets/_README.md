Infos about the models and tokenizers:

* In general: The files are named by MODELNAME_TOKENIZERNAME.csv
* 4 tokenizers were used:
  * character level - 'char'
  * word level - 'word'
  * gpt2 pretrained (by huggingface) - 'gpt2'
  * gpt2 from scratch - 'gpt2-trained'
  
<br>

* There is also the gpt2 transformer model, which was trained in three ways:
    * fine tuned: just fine tuned the pretrained the huggingface model on musk data
    * net scratch: trained the net from scratch but kept the pretrained tokenizer
    * tokenizer scratch: trained the tokenizer and model from scratch on elon data