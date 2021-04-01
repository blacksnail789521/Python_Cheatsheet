import numpy as np


array_1 = np.array([0, 1, 2])
array_2 = np.array([[0, 1], [2, 3]])

np.savez("test.npz", array_1 = array_1, array_2 = array_2)
# np.savez_compressed("test.npz", array_1 = array_1, array_2 = array_2)

loaded = np.load("test.npz")
print("array_1:", loaded["array_1"])
print("array_2:", loaded["array_2"])