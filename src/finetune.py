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
            lines.append(line)
        return lines


input_data = read_lines("./data/input_examples.json")
output_data = read_lines("./data/output_examples.json")
training_data = []

# for i in range(0, len(input_data) - 3, 3):
#     size = random.randint(a=1, b=3)
#     if size == 1:
#         concat_input = json.loads(input_data[i])
#         concat_output = json.loads(output_data[i])
#     elif size == 2:
#         concat_input = json.loads(input_data[i]) + json.loads(input_data[i + 1])
#         concat_output = json.loads(output_data[i]) + json.loads(output_data[i + 1])
#     elif size == 3:
#         concat_input = (
#             json.loads(input_data[i])
#             + json.loads(input_data[i + 1])
#             + json.loads(input_data[i + 2])
#         )
#         concat_output = (
#             json.loads(output_data[i])
#             + json.loads(output_data[i + 1])
#             + json.loads(output_data[i + 2])
#         )
#     else:
#         raise ValueError("randint generated an invalid number")
#
#     training_data.append(
#         {
#             "text_input": json.dumps(concat_input),
#             "output": json.dumps(concat_output),
#         }
#     )
#
# for i, (input_line, output_line) in enumerate(zip(input_data, output_data)):
#     if i == 0 or i % 2 == 0:
#         concat_input = json.loads(input_data[i]) + json.loads(input_data[i + 1])
#         concat_output = json.loads(output_data[i]) + json.loads(output_data[i + 1])
#         training_data.append(
#             {
#                 "text_input": json.dumps(concat_input),
#                 "output": json.dumps(concat_output),
#             }
#         )

# data = read_lines("prompt_examples.json")
#
# for i, line in enumerate(data):
#     if i == 0 or i % 2 == 0:
#         training_data.append(
#             {"text_input": data[i].strip(), "output": data[i + 1].strip()}
#         )
#     else:
#         continue
#

with open("prompt.md") as file:
    prompt = file.read()

    for input_line, output_line in zip(input_data, output_data):
        training_data.append(
            {
                "text_input": prompt + "\n" + input_line.strip(),
                "output": output_line.strip(),
            }
        )


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


for model_info in genai.list_tuned_models():
    print(model_info)

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    display_name="gemini-flash-tune-eval",
    source_model=base_model,
    epoch_count=25,
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

print(result)
