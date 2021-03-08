import numpy as np


A = np.array([[1,2,3],
              [4,5,6]])
B = np.array([[1,2,3],
              [4,5,6]])

a = np.array([1,2,3])
b = np.array([1,2,3])

print("-----------------")
print("inner product")
print(np.dot(a, b))

print("-----------------")
print("element-wise product")
print(np.multiply(A, B))
print(a * b)


A = np.array([[1,2,3],
              [4,5,6]])

B = np.array([[1,2],
              [3,4],
              [5,6]])

print("-----------------")
print("matrix multiplication")
print(np.matmul(A, B))
print(A @ B)
