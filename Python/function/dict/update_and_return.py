a = {"A": 1, "B": 2}
to_be_merged = {"C": 3}
print(dict(a, **to_be_merged))