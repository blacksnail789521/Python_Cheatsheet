from collections import OrderedDict


a = OrderedDict()
a["key_3"] = {"XD": 1}
a["key_2"] = {"XD": 2}
a["key_1"] = {"XD": 3}

# key.
print("--------------------------------------")
for key in a.keys():
    print(f"key: {key}")

# value.
print("--------------------------------------")
for value in a.values():
    print(f"value: {value}")

# key and value.
print("--------------------------------------")
for key, value in a.items():
    print(f"key: {key}, value: {value}")

# index, key, value.
print("--------------------------------------")
for index, (key, value) in enumerate(a.items()):
    print(f"index: {index}, key: {key}, value: {value}")