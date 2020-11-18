import numpy as np


m, n = 2, 3

print("-----------------")
print("Without initializing values.")
a = np.empty((m, n))
print(a)

print("-----------------")
print("Filled with zeros")
a = np.zeros((m, n))
print(a)

print("-----------------")
print("Filled with ones")
a = np.ones((m, n))
print(a)

print("-----------------")
print("Filled with given values.")
a = np.full((m, n), np.inf)
print(a)