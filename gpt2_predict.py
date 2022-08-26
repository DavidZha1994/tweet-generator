from transformers import GPT2Tokenizer, pipeline, GPT2LMHeadModel, GPT2Config
from pathlib import Path
import pandas as pd
from utils import get_prompts

selection = "gpt2_tokenizer_scratch"
input_prompts = get_prompts()
returned_seqs = 1

options = {
    "gpt2_fine_tuned": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "dir_name": "fine_tuned"
    },
    "gpt2_net_scratch": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "dir_name": "from_scratch"
    },
    "gpt2_tokenizer_scratch": {
        "tokenizer": GPT2Tokenizer('gpt2tokenizer/vocab.json', 'gpt2tokenizer/merges.txt'),
        "dir_name": "token_trained"
    },
    "gpt2_one_ep": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "dir_name": "fine_tuned_one_ep"
    }
}

tokenizer = options[selection]["tokenizer"]
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_json_file(f"./models/{options[selection]['dir_name']}/config.json")
model = GPT2LMHeadModel.from_pretrained(f"./models/{options[selection]['dir_name']}")

generate = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

generation_outputs = generate(input_prompts, max_length=30, num_return_sequences=returned_seqs)

file_path = f"./sample_generated_tweets/{selection}.csv"
file = Path(file_path)
file.parent.mkdir(exist_ok=True, parents=True)

outputs = []
prompts = []
for output, prompt in zip(generation_outputs, input_prompts):
    outputs.extend(output)
    prompts.extend([prompt] * returned_seqs)

data = [[prompt, output['generated_text']] for prompt, output in zip(prompts, outputs)]
df = pd.DataFrame(data=data, columns=['prompt', 'output'])

df.to_csv(file_path, mode="w+")

print(generation_outputs)
