import numpy as np


numerator   = np.array([0, 1, 0, 3])
denominator = np.array([0, 1, 2, 0])

# Substitute with 0.
ratio_list = np.divide( numerator, denominator, where = (denominator != 0), out = np.array([0] * len(numerator), dtype = float) )
print(ratio_list)

# Substitute with np.nan.
ratio_list = np.divide( numerator, denominator, where = (denominator != 0), out = np.array([np.nan] * len(numerator)) )
print(ratio_list)