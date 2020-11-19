import pandas as pd


print("=================================")
df = pd.DataFrame({"A": [0, 1], "B": ["2", "3"], \
                  "datetime": ["2020/12/01 08:09:10", "2020/12/01 08:09:07"]})
print(df)
print("---------------")
print(df.dtypes)

# numerical.
print("=================================")
print("Change to numerical:")
df["B"] = df["B"].astype(int)
print(df)
print("---------------")
print(df.dtypes)

# categorical.
print("=================================")
print("Change to categorical:")
df["A"] = df["A"].astype(object)
print(df)
print("---------------")
print(df.dtypes)

# datetime.
print("=================================")
print("Change to datetime:")
df["datetime"] = pd.to_datetime(df["datetime"], format = "%Y/%m/%d %H:%M:%S.%f")
print(df)
print("---------------")
print(df.dtypes)