import pandas as pd
import numpy as np


print("----------------------")
df = pd.DataFrame(
    {
        "A": [1, np.nan, 7, np.nan],
        "B": [np.nan, 5, 8, 8],
        "C": [np.nan, 3, np.nan, np.nan],
    },
    dtype=float,
)
print(df)

print("----------------------")
print("fill with 0")
print(df.fillna(0))

print("----------------------")
print("fill with forward propagate")
print(df.fillna(method="ffill"))

print("----------------------")
print("fill with back propagate")
print(df.fillna(method="bfill"))

print("----------------------")
print("fill with interpolation")
print(df.interpolate())

print("----------------------")
print("fill with interpolation (without any NaN value left)")
print(df.interpolate().fillna(method="bfill"))
