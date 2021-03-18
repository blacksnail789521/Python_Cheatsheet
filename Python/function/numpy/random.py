import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

print("----------------------------")
print("Generate 5 int from 1 to 10.")
print( np.random.randint(1, 10, 5) )

print("----------------------------")
print("Generate 5 float with \"0 mean, 1 std\" using normal distribution.")
print( np.random.randn(5) )

print("----------------------------")
print("Generate 5 float from 0 to 1.")
print( np.random.random(5) )

print("----------------------------")
print("Generate 5 float from -10 to 10.")
print( np.random.uniform(-10, 10, 5) )

print("----------------------------")
print("Generate 10000 float from 0.1 to 0.0001 in log scale.")
random_list = sorted(10**np.random.uniform(-4, -1, 10000))
plt.plot(random_list)
plt.yscale('log')
plt.show()