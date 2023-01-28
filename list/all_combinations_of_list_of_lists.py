import itertools

list_of_lists = [[4, 8, 16], ["a", "b"]]

# Get all combinations of a list of lists
products = list(itertools.product(*list_of_lists))
print(products)  # [(4, 'a'), (4, 'b'), (8, 'a'), (8, 'b'), (16, 'a'), (16, 'b')]
