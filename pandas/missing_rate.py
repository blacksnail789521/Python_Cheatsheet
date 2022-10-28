import pandas as pd
import numpy as np


print("----------------------")
df = pd.DataFrame({"A": [1, np.nan, 7], "B": [2, 5, 8], "C": [3, np.nan, np.nan]}, dtype = int)
print(df)

print("----------------------")
print("Drop columns with missing_rate over 0.5")
df = df.loc[:, df.isnull().mean() < 0.5]
print(df)