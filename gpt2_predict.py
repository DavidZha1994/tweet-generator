from transformers import GPT2Tokenizer, pipeline, GPT2LMHeadModel, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_json_file("./models/fine_tuned/config.json")
model = GPT2LMHeadModel.from_pretrained("./models/fine_tuned")

generate = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

tx = generate("", max_length=30, num_return_sequences=10000)
print(tx)