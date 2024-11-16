import json
import os

from transformers import TextStreamer
from unsloth import FastLanguageModel

max_seq_length = 256
dtype = None
load_in_4bit = False

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="review-model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)


def read_prompt():
    with open("test_prompt.txt") as data:
        with open("../prompt.md") as prompt:
            return prompt.read() + "\n" + data.read()


prompt = read_prompt()
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
tensor_response = model.generate(
    input_ids,
    streamer=text_streamer,
    pad_token_id=tokenizer.eos_token_id,
)

response = tokenizer.batch_decode(tensor_response)[0]
print(response)

before, match, after = response.partition("### Response:")
if match:
    result = match + after
else:
    raise ValueError("Failed to parse response.")

result_lines = result.splitlines()[2:-2]
result_json = json.loads("".join(result_lines))
print(result_json)
