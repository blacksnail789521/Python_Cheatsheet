import pandas as pd


df = pd.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})
print(df)

# Use dict.
print("---------------")
df.loc[len(df)] = {"A": 7, "B": 8, "C": 9}
df.loc[len(df)] = {"B": 11, "C": 12, "A": 10}
print(df)

# Use list.
print("---------------")
df.loc[len(df)] = [13, 14, 15]
print(df)