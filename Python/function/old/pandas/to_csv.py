import pandas as pd

df = pd.DataFrame({ "WAFER_ID": [1, 2, 3, 5],
                    "value": ["foo", "bar", "baz", "foo"] })
print(df)
df.to_csv("test.csv", index = False)

# It's the same.
#df.to_csv("test.csv", index = False, sep = ",")