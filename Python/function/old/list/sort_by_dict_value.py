a = [{'name':'Homer', 'age':39}, {'name':'Bart', 'age':10}]
print(a)

a = sorted(a, key=lambda k: k['name']) 
print(a)