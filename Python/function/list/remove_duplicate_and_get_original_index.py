from collections import OrderedDict


a = ["A", "B", "B", "C", "B"]

# Use element as key in dict.
d = OrderedDict()
for index, element in enumerate(a):
    if d.get(element, None) is None:
        d[element] = index
print( list(d.keys()) )
print( list(d.values()) )