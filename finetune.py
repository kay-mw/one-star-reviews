import os
import time
from typing import List

import google.generativeai as genai
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from typing_extensions import Generator


def read_lines(path: str) -> List[str]:
    with open(path, "r") as file:
        lines = []
        for line in file:
            lines.append(line)
        return lines


# input_data = read("input_examples.json")
# output_data = read("output_examples.json")
data = read_lines("prompt_examples.json")

training_data = []
for i, line in enumerate(data):
    if i == 0 or i % 2 == 0:
        training_data.append(
            {"text_input": data[i].strip(), "output": data[i + 1].strip()}
        )
    else:
        continue

# for inputs, outputs in zip(input_data, output_data):
#     training_data.append({"text_input": inputs.strip(), "output": outputs.strip()})


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    display_name="synthetic-data-1.0",
    source_model=base_model,
    epoch_count=10,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()

snapshots = pd.DataFrame(result.tuning_task.snapshots)
fig = px.scatter(snapshots, x="epoch", y="mean_loss", template="plotly_dark")
fig.show()
