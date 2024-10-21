import json

with open("prompt_examples.json", "r") as file:
    contents = file.read()
    contents = json.loads(contents)

final = []
for i, content in enumerate(contents):
    if i == 0:
        final.append(contents[0:10])
    else:
        start = i * 10
        final.append(contents[start : start + 10])
        if start >= len(contents) - 10:
            break

with open("input_examples.json", "w") as file:
    for category in final:
        file.write(json.dumps(category) + "\n")
