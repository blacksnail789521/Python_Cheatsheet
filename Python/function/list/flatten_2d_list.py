import itertools


list2d = [ [1, 2, 3], [4, 5, 6], [7], [8, 9] ]
flattened_list_1 = [ item for sublist in list2d for item in sublist ]
flattened_list_2 = list(itertools.chain.from_iterable(list2d))

assert flattened_list_1 == flattened_list_2