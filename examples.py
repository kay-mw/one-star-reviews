import json
import pprint


def read_lines(path: str):
    with open(path, "r") as file:
        for line in file:
            yield line


line_generator = read_lines("prompt_examples.json")
base = []
for line in line_generator:
    review_dict = json.loads(line)
    timestamps = []
    for entry in review_dict:
        pprint.pprint(entry)
        try:
            score = int(input("Score: "))
        except ValueError:
            score = int(input("Try again: "))
        print("\n")
        timestamps.append({"timestamp": entry["timestamp"], "score": score})

    base.append(timestamps)


print(base)

open("base.json", "w").close()
with open("base.json", "a") as file:
    for b in base:
        file.write(str(b) + "\n")
