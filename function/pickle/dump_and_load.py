import pickle
import pandas as pd


before = {"1": "XD", "2": "HAHA", "3": pd.DataFrame()}

# Dump.
with open("data.pickle", "wb") as pickle_file:
    pickle.dump(before, pickle_file, pickle.HIGHEST_PROTOCOL)

# Load.
with open("data.pickle", "rb") as pickle_file:
    after = pickle.load(pickle_file)

print(before)
print(after)