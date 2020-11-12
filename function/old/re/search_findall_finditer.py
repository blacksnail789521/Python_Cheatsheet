import re

event_name = "CH_[10-21]WEX01_OCCUPIED"

# search can get first match and its index.
print("search:")
element = re.search(r"\d+", event_name)
print(element.group())
print(element.start(), element.end())

# findall can get all matches.
print("findall:")
for element in re.findall(r"\d+", event_name):
    print(element)

# finditer can get all matches and its index.
print("finditer:")
for element in re.finditer(r"\d+", event_name):
    start_index, end_index = element.start(), element.end()
    print(event_name[start_index : end_index])