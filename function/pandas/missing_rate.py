import pandas as pd
import numpy as np


print("----------------------")
df = pd.DataFrame({"A": [1, np.nan, 7], "B": [2, 5, 8], "C": [3, np.nan, np.nan]}, dtype = int)
print(df)

print("----------------------")
missing_rate_series = df.isnull().mean()
print(f"missing_rate_series: \n{missing_rate_series}")

print("----------------------")
drop_column_name_list = missing_rate_series[missing_rate_series > 0.5].index.tolist()
print(f"drop_column_name_list: {drop_column_name_list}")