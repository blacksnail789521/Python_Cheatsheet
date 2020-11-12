import pandas as pd

df1 = pd.DataFrame({ "WAFER_ID": [1, 2, 3, 5],
                     "value": ["foo", "bar", "baz", "foo"] })
df2 = pd.DataFrame({ "WAFER_ID": [5, 6, 7, 8],
                     "value": ["foo", "bar", "baz", "foo"] })

print(df1)
print(df2)

df_list = [df1, df2]

# Use list to concat (Remember to set ignore_index).
df = pd.concat( df_list, axis = 1 )
print(df)