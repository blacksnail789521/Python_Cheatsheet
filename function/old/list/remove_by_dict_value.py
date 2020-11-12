a = [{'name':'Homer', 'age':"X"}, {'name':'Bart', 'age':10}]
print(a)

a = [d for d in a if d.get('age') != "X"]
print(a)