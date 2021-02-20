import pandas as pd
import numpy as np


print("----------------------")
df = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, np.nan, np.nan], "C": [3, np.nan, np.nan]}, dtype = int)
print(df)

print("----------------------")
print("Drop columns with single value (It won't drop the columns with all NAN.)")
df = df.loc[:, df.apply(pd.Series.nunique) != 1]
print(df)