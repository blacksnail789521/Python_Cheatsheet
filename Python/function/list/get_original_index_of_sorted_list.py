# index:     0   1  2  3  4
some_list = [9, -8, 4, 2, 2]

# Sort index_list based on corresponding value in list.
original_index_list = sorted( range(len(some_list)), key = lambda index : some_list[index] )
print(original_index_list)