import json


before = {"1": "XD", "2": "HAHA"}

# Dump.
with open("data.json", "w") as json_file:
    json.dump(before, json_file, indent = 4)

# Load.
with open("data.json", "r") as json_file:
    after = json.load(json_file)

assert before == after