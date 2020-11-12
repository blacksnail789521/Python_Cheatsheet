import re


test_string = "CH_[10_21]"
print(f"test_string: {test_string}")
print("index:      ", "".join([ str(i) for i in range(len(test_string)) ]))

print("-------------------------------------------")
print("\"serach\" can get first match and its index:")
element = re.search(r"\d+", test_string)
print("{number} (start_index: {start_index}, end_index: {end_index})" \
      .format(number = element.group(), start_index = element.start(), end_index = element.end()))

print("-------------------------------------------")
print("\"findall\" can get all matches:")
for element in re.findall(r"\d+", test_string):
    print(element)

print("-------------------------------------------")
print("\"finditer\" can get all matches and their indices:")
for element in re.finditer(r"\d+", test_string):
    print("{number} (start_index: {start_index}, end_index: {end_index})" \
          .format(number = element.group(), start_index = element.start(), end_index = element.end()))