from transformers import pipeline, RobertaTokenizerFast, RobertaForMaskedLM
import language_tool_python

USE_PRETRAINED = False

tokenizer_weights = 'roberta-base' if USE_PRETRAINED else './elberta/2'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_weights, max_len=512)

model = RobertaForMaskedLM.from_pretrained("roberta-base") if USE_PRETRAINED else './elberta/2'


fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

seq = "Women only want"
for _ in range(10):
    fill = fill_mask(seq + " <mask>")
    seq += fill[0]['token_str']
    print(seq)

tool = language_tool_python.LanguageTool('en-US')
matches = tool.check(seq)
print(matches)
print(len(matches))
tool.close()