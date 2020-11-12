from collections import OrderedDict


# Use array-like object of tuple to initialize.
a = OrderedDict([("b", 2), ("a", 1)])
print(a)

a = OrderedDict((("b", 2), ("a", 1)))
print(a)