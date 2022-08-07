from transformers import GPT2Tokenizer, pipeline, GPT2LMHeadModel, Trainer, TrainingArguments, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, TrainerCallback, TrainerControl, TrainerState
from tokenizers.implementations import ByteLevelBPETokenizer
import math
from pathlib import Path

DEVICE = 'cuda:0'
TRAIN_NN_FROM_SCRATCH = False
TRAIN_TOKENIZER_FROM_SCRATCH = False

if TRAIN_NN_FROM_SCRATCH == False and TRAIN_TOKENIZER_FROM_SCRATCH == False:
    save_name = 'fine_tuned'
if TRAIN_NN_FROM_SCRATCH == True and TRAIN_TOKENIZER_FROM_SCRATCH == False:
    save_name = 'nn_trained'
if TRAIN_NN_FROM_SCRATCH == True and TRAIN_TOKENIZER_FROM_SCRATCH == True:
    save_name = 'token_nn_trained'
    
# disallow the possibility to train the tokenizer without retraining the model
assert TRAIN_NN_FROM_SCRATCH or not TRAIN_TOKENIZER_FROM_SCRATCH, "Set TRAIN_NN to true if TRAIN_TOKENIZER=true"

class EvalCallback(TrainerCallback):

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Reference: https://github.com/huggingface/transformers/blob/11c69b80452fae4b13c6d8bc22bdc19f3a752199/examples/pytorch/language-modeling/run_clm.py#L494
        metrics = kwargs['metrics']
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", state.log_history, combined=False)
        print(f"{perplexity=:.2f}")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model_conf = model.config

if TRAIN_TOKENIZER_FROM_SCRATCH:
    paths = str(Path('./dataset/combined_Musk_tweets_cleaned.txt'))

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=paths, vocab_size=10_000, min_frequency=2, special_tokens=[
        '<|endoftext|>'
    ])

    tokenizer.save_model("gpt2tokenizer")

    model_conf.vocab_size = 10_000
    tokenizer = GPT2Tokenizer('gpt2tokenizer/vocab.json', 'gpt2tokenizer/merges.txt')
    tokenizer.pad_token = tokenizer.eos_token

if TRAIN_NN_FROM_SCRATCH:
    model = GPT2LMHeadModel(model_conf)

train_set = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./dataset/train_cleaned.txt",
    block_size=32,
)

test_set = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./dataset/test_cleaned.txt",
    block_size=32,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir=f"./metrics/{save_name}",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    save_steps=5_000,
    save_total_limit=2,
    prediction_loss_only=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to='tensorboard'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=test_set,
    callbacks=[EvalCallback]
)

trainer.train()

generate = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

trainer.save_model(f"./models/{save_name}")

tx = generate("Tesla is", max_length=30, num_return_sequences=5)
print(tx)
