import pandas as pd


# Append a new column and assign it as index.
print("------------------")
df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
df["index"] = [-1, -2, -3]
df = df.set_index("index")
print(df)

# Use "pd.Index" directly.
print("------------------")
df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
df = df.set_index(pd.Index([-1, -2, -3], name = "index"))
print(df)