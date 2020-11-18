import numpy as np


np_2darray = np.array([[0, 1], [2, 3]])
print(f"np_2darray: \n{np_2darray}")

# Concatenate some rows.
print("------------------------------")
print("Concatenate 2 rows.")
new_array = np.array([[4, 5], [6, 7]])
np_2darray = np.vstack((np_2darray, new_array))
# np_2darray = np.concatenate((np_2darray, new_array), axis = 0)
print(f"np_2darray: \n{np_2darray}")

# Concatenate some cols.
print("------------------------------")
print("Concatenate 2 cols.")
new_array = np.array([[-1, -1, -1, -1], [-2, -2, -2, -2]]).T
np_2darray = np.hstack((np_2darray, new_array))
# np_2darray = np.concatenate((np_2darray, new_array), axis = 1)
print(f"np_2darray: \n{np_2darray}")

# Drop some rows.
print("------------------------------")
print("Drop the first and third rows.")
np_2darray = np.delete(np_2darray, [0, 2], axis = 0)
print(f"np_2darray: \n{np_2darray}")

# Drop some cols.
print("------------------------------")
print("Drop the first and third cols.")
np_2darray = np.delete(np_2darray, [0, 2], axis = 1)
print(f"np_2darray: \n{np_2darray}")