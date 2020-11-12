import pandas as pd

df = pd.DataFrame({ "WAFER_ID": [1, 5, 7, 4],
                    "value": ["foo1", "bar", "baz", "foo2"] })

print(df)

# Based on length.
#df = df.loc[ df["value"].map(len) < 4 ].reset_index(drop = True)

# Based on value.
df = df.loc[ df["WAFER_ID"] < 5, ["value"] ].reset_index(drop = True)
print(df)