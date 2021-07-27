import itertools

list_of_lists = [[1,2],[3,4]]

# Get all combinations of a list of lists
products = list(itertools.product(*list_of_lists)) # [(1, 3), (1, 4), (2, 3), (2, 4)]