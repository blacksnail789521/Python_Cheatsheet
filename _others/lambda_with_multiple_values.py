def test(a, b):
    return 2*a, 2*b
print(test(1, 2))
'''
(2, 4)
'''

test_lambda = lambda a, b: (2*a, 2*b) # Only the return part needs to add ()
print(test_lambda(1, 2))
'''
(2, 4)
'''