import pandas as pd
import numpy as np


df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [30, np.nan, np.nan]}, \
                  index = pd.Index([-1, -2, -3], name = "index"), dtype = int)
print(df)

# Select rows by index's label.
print("====================================================")
print("Select the rows with index label == -1:")
print("---------------------")
print("### series: ###")
print(df.loc[-1])
print("---------------------")
print("### df: ###")
print(df.loc[[-1]])

# Select rows by index's location.
print("====================================================")
print("Select the rows with index location == 0:")
print("---------------------")
print("### series: ###")
print(df.iloc[0])
print("---------------------")
print("### df: ###")
print(df.iloc[[0]])
print("---------------------")
print("### df: (using [] without loc and iloc, similar with np.array) ###")
print(df[0:1]) # Couldn't use without slicing. (ex: df[0])

# Select rows by rules.
print("====================================================")
print("Select the rows with (1) odd index labels (2) A > 3:")
print(df.loc[ (df.index % 2 == 1) & (df["A"] > 3) ])

print("---------------------")
print("Do NOT select the rows with (1) odd index labels (2) A > 3:")
print(df.loc[ ~ ((df.index % 2 == 1) & (df["A"] > 3)) ])

# Select columns by column name.
print("====================================================")
print("Select the columns with column name == A:")
print("---------------------")
print("### series: ###")
print(df["A"])
print("---------------------")
print("### df: ###")
print(df[["A"]])

# Select columns by column names' rules.
print("====================================================")
print("Select the columns with re.search(r\"A\", column_name) == True:")
print(df.loc[ : , df.columns.str.contains(r"A", regex = True) ])

# Select columns by column values' rules.
print("====================================================")
print("Select the columns with (1) sum of value > 14 (2) missing rate < 50%:")
print(df.loc[ : , (df.sum() > 14) & (df.isnull().mean() < 0.5) ])