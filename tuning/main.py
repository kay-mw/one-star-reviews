import json
import os
from time import gmtime, strftime

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import (
    FastLanguageModel,
    apply_chat_template,
    is_bfloat16_supported,
    standardize_sharegpt,
    to_sharegpt,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["LLAMA_CPP_LIB"] = "./llama.cpp"

max_seq_length = 512
dtype = None
load_in_4bit = True
r = 16
lora_alpha = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_alpha,
    lora_dropout=0,
    use_gradient_checkpointing=True,
    random_state=1,
    use_rslora=False,
    loftq_config=None,
)


def prep_data() -> Dataset:
    def read_lines(path: str):
        with open(path, "r") as file:
            for line in file:
                yield json.loads(line.strip())

    input_data = read_lines("./data/input_examples.json")
    output_data = read_lines("./data/output_examples.json")

    examples = []
    for input_line, output_line in zip(input_data, output_data):
        example = {
            "instruction": "Score the following product reviews. Output only a JSON array containing timestamp and score pairs.",
            "input": f"{input_line}",
            "output": f"{output_line}",
        }
        examples.append(example)

    dataset = Dataset.from_list(examples)

    return dataset


dataset = prep_data()
dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=1,
)
dataset = standardize_sharegpt(dataset)

chat_template = """Below is some review data. Score each of these product reviews by outputting a JSON array containing timestamp and score pairs.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

model.save_pretrained_gguf(
    f"{strftime("%H-%M", gmtime())}-model",
    # "15-06-model",
    tokenizer,
    maximum_memory_usage=0.2,
    # save_method="merged_16bit",
)
