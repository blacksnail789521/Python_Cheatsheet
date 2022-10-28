import pandas as pd


df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
print(df)

print("---------------")
df = df.sort_values(["C", "B"], ascending = False).reset_index(drop = True)
print(df)