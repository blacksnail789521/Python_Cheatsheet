import numpy as np


# 1d_list and 1d_array.
a = [0, 1, 3]
np_a = np.array(a)
print(f"a: {a}")
print(f"np_a: {np_a}")

# Append.
element = 4
print("-------------------------------------")
print(f"Append {element}.")
a.append(element)
np_a = np.append(np_a, element)
print(f"a: {a}")
print(f"np_a: {np_a}")

# Insert.
index, element = 2, -2
print("-------------------------------------")
print(f"Insert {element} at {index}.")
a.insert(index, element)
np_a = np.insert(np_a, index, element)
print(f"a: {a}")
print(f"np_a: {np_a}")

# Extend.
list_ = [5, 6, 5, 5, 7]
print("-------------------------------------")
print(f"Extend a list {list_}.")
a.extend(list_)
np_a = np.append(np_a, list_)
# np_a = np.concatenate((np_a, list_))
# np_a = np.hstack((np_a, list_))
print(f"a: {a}")
print(f"np_a: {np_a}")

# Remove the first one.
drop_element = 5
print("-------------------------------------")
print(f"Remove the first encountered {drop_element}.")
a.remove(drop_element)
np_a = np.delete(np_a, drop_element)
print(f"a: {a}")
print(f"np_a: {np_a}")

# Remove all.
drop_element = 5
print("-------------------------------------")
print(f"Remove every encountered {drop_element}.")
a = [ element for element in a if element != drop_element ]
np_a = np.delete(np_a, np.where(np_a == drop_element))
print(f"a: {a}")
print(f"np_a: {np_a}")