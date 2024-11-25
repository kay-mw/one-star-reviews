import json
import os
import time
from typing import List

import google.generativeai as genai
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv


def read_lines(path: str) -> List[str]:
    with open(path, "r") as file:
        lines = []
        for line in file:
            lines.append(json.loads(line))
        return lines


input_data = read_lines("./data/new_input_examples.json")
output_data = read_lines("./data/new_output_examples.json")
training_data = []

with open("prompt.md") as file:
    prompt = file.read()

    for input_line, output_line in zip(input_data, output_data):
        training_data.append(
            {
                "text_input": prompt + "\n" + json.dumps(input_line).strip(),
                "output": json.dumps(output_line).strip(),
            }
        )

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


for model_info in genai.list_tuned_models():
    print(model_info)

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    display_name="gemini-flash-tune-ordered",
    source_model=base_model,
    epoch_count=22,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)


operation.metadata

result = operation.result()

snapshots = pd.DataFrame(result.tuning_task.snapshots)
fig = px.scatter(snapshots, x="epoch", y="mean_loss", template="plotly_dark")
fig.show()

print(result)
