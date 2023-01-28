import itertools
from pprint import pprint

batch_size_list = [4, 8, 16]
optimizer_list = ["Adam", "SGD"]

# Get all combinations of a list of lists
all_combinations = list(itertools.product(*[batch_size_list, optimizer_list]))
print(all_combinations)
"""
[(4, 'Adam'), (4, 'SGD'), (8, 'Adam'), (8, 'SGD'), (16, 'Adam'), (16, 'SGD')]
"""

for i, (batch_size, optimizer) in enumerate(all_combinations):
    print(f"i = {i}, batch_size = {batch_size}, optimizer = {optimizer}")
"""
i = 0, batch_size = 4, optimizer = Adam
i = 1, batch_size = 4, optimizer = SGD
i = 2, batch_size = 8, optimizer = Adam
i = 3, batch_size = 8, optimizer = SGD
i = 4, batch_size = 16, optimizer = Adam
i = 5, batch_size = 16, optimizer = SGD
"""

"""--------------------------------------------------------------------"""

# Get all combinations of a dictionary of lists (keep the names)
my_dict = {"batch_size": batch_size_list, "optimizer": optimizer_list}
keys, values = zip(*my_dict.items())
all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
pprint(all_combinations)
"""
[{'batch_size': 4, 'optimizer': 'Adam'},
 {'batch_size': 4, 'optimizer': 'SGD'},
 {'batch_size': 8, 'optimizer': 'Adam'},
 {'batch_size': 8, 'optimizer': 'SGD'},
 {'batch_size': 16, 'optimizer': 'Adam'},
 {'batch_size': 16, 'optimizer': 'SGD'}]
"""

for i, combination in enumerate(all_combinations):
    print(f"i = {i}, combination = {combination}")
"""
i = 0, combination = {'batch_size': 4, 'optimizer': 'Adam'}
i = 1, combination = {'batch_size': 4, 'optimizer': 'SGD'}
i = 2, combination = {'batch_size': 8, 'optimizer': 'Adam'}
i = 3, combination = {'batch_size': 8, 'optimizer': 'SGD'}
i = 4, combination = {'batch_size': 16, 'optimizer': 'Adam'}
i = 5, combination = {'batch_size': 16, 'optimizer': 'SGD'}
"""

# We can use an element of all_combinations to update a dictionary
hp_dict = {"batch_size": None, "optimizer": None}
hp_dict.update(all_combinations[0])
print(hp_dict)
"""
{'batch_size': 4, 'optimizer': 'Adam'}
"""
