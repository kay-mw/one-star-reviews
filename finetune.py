import os
import time

import google.generativeai as genai
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from typing_extensions import Generator


def read(path: str) -> Generator:
    with open(path, "r") as file:
        for line in file:
            yield line


input_data = read("input_examples.json")
output_data = read("output_examples.json")

training_data = []
for inputs, outputs in zip(input_data, output_data):
    training_data.append({"text_input": inputs.strip(), "output": outputs.strip()})


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    display_name="increment",
    source_model=base_model,
    epoch_count=20,
    batch_size=len(training_data),
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()

snapshots = pd.DataFrame(result.tuning_task.snapshots)
fig = px.line(snapshots, x="epoch", y="mean_loss", template="plotly_dark")
fig.show()

with open("finetune_names.txt", "a") as file:
    file.write(f"Run 1: {result.name}" + "\n")
