import pickle
import pandas as pd


before = {"1": "XD", "2": 2, "3": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}

# Dump.
with open("data.pkl", "wb") as pickle_file:
    pickle.dump(before, pickle_file, pickle.HIGHEST_PROTOCOL)

# Load.
with open("data.pkl", "rb") as pickle_file:
    after = pickle.load(pickle_file)

print(before)
print(after)