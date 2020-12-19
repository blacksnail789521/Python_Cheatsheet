import pandas as pd

df = pd.DataFrame({ "WAFER_ID": [1, 5, 7, 1],
                    "value": ["foo1", "bar", "baz", "foo2"] })

print(df)

for index, row in df.iterrows():
    if row["WAFER_ID"] == 1:
        df.loc[index, "value"] = "XD"

print(df)