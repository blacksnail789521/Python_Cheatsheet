import pandas as pd


print("----------------------")
df = pd.DataFrame(
    {"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]},
    index=pd.Index(["a", "b", "c"], name="index"),
)
print(df)
"""
       A  B  C
index         
a      1  2  3
b      4  5  6
c      7  8  9
"""

# Rename column name
print("----------------------")
new_df = df.rename(columns={"A": "A1", "B": "B2", "C": "C3"})
print(new_df)
"""
       A1  B2  C3
index            
a       1   2   3
b       4   5   6
c       7   8   9
"""

# Rename index name
print("----------------------")
new_df = df.rename_axis("new_index")
print(new_df)
"""
            A  B  C
new_index
a           1  2  3
b           4  5  6
c           7  8  9
"""

# Rename index label
print("----------------------")
new_df = df.rename(index={"a": "a1", "b": "b2", "c": "c3"})
print(new_df)
"""
       A  B  C
index         
a1     1  2  3
b2     4  5  6
c3     7  8  9
"""

print()
