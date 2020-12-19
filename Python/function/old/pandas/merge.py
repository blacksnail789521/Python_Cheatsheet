import pandas as pd

df1 = pd.DataFrame({ "WAFER_ID": [1, 2, 3, 5],
                     "value": ["foo", "bar", "baz", "foo"] })
df2 = pd.DataFrame({ "WAFER_ID": [5, 6, 7, 8],
                     "value": ["foo", "bar", "baz", "foo"] })

print(df1)
print(df2)

df1 = pd.merge(df1, df2, on = "WAFER_ID", how = "inner")
print(df1)


# Not faster version.
#df1 = df1.set_index("WAFER_ID").join(df2.set_index("WAFER_ID"), how = "inner", lsuffix = "_x", rsuffix = "_y").reset_index()
#print(df1)