import pandas as pd

print("---------------")
df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["A", "B", "C", "D"]}, \
                  index = pd.Index([-1, -2, -3, -4], name = "index"))
print(df)

# Shuffle
print("---------------")
df = df.sample(frac=1)
print(f"df: \n{df}")