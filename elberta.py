from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, RobertaTokenizerFast, LineByLineTextDataset, TrainerCallback, TrainerState, \
    TrainerControl, pipeline

USE_PRETRAINED = False

paths = str(Path('./dataset/combined_Musk_tweets_cleaned.txt'))

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=10000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("elberta/2")

tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

tokenizer.enable_truncation(max_length=512)

config = RobertaConfig(
    vocab_size=10000,
    max_position_embeddings=256,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer_weights = 'roberta-base' if USE_PRETRAINED else './elberta/2'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_weights, max_len=512)

model = RobertaForMaskedLM.from_pretrained("roberta-base") if USE_PRETRAINED else RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./dataset/train_cleaned.txt",
    block_size=64,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./check/2",
    overwrite_output_dir=True,
    num_train_epochs=300,
    per_gpu_train_batch_size=128,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
)


class EvalCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.pred(kwargs['model'])

    def pred(self, model):
        fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer="./elberta/2",
            device=0
        )

        seqs = ["Tesla", "After", "Why"]
        for _ in range(10):
            for idx, seq in enumerate(seqs):
                fill = fill_mask(seq + " <mask>")
                seqs[idx] = seq + fill[0]['token_str']

        print(seqs)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # callbacks=[EvalCallback]
)

trainer.train()
trainer.save_model("./elberta/2")
