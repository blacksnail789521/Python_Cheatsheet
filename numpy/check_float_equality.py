import numpy as np


a = 0.7
b = 0.3

np.testing.assert_allclose(sum([a, b]), 1)
np.testing.assert_allclose(sum([a, b]), 0.99999)