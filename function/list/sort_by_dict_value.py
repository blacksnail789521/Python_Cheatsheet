from pprint import pprint


a = [{"name": "Brain", "age": 32, "salary": 50000}, \
     {"name": "Carol", "age": 27, "salary": 60000}, \
     {"name": "Jason", "age": 27, "salary": 20000}]
print("before:")
pprint(a)

a = sorted(a, key = lambda d : (d["age"], d["salary"]), reverse = False) # Default is ascending (reverse = False)
print("after:")
pprint(a)