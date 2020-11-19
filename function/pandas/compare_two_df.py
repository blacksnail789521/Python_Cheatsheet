import pandas as pd


df1 = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["A", "B", "C", "D"]})
df2 = pd.DataFrame({"A": [4, 3, 1, 2], "B": ["D", "C'", "A", "B'"]})
df2_with_order = df2.sort_values(["A"]).reset_index(drop = True)

# Compare with considering order.
diff_with_order = df1.compare(df2_with_order)

# Compare without considering order.
diff_without_order = pd.concat({"df1": df1, "df2": df2}).drop_duplicates(keep = False).reset_index() \
                         .rename(columns = {"level_0": "df name", "level_1": "row index"})