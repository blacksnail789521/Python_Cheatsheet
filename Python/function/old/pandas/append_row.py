import pandas as pd

df = pd.DataFrame({ "WAFER_ID": [1, 5, 7, 4],
                    "value": ["foo1", "bar", "baz", "foo2"] })

print(df)

# Add this new row into row_for_ratio_table.
if pd.isnull(df.index.max()) == True:
    df_index = 0
else:
    df_index = df.index.max() + 1
df.loc[ df_index ] = [9, "XD"]

print(df)