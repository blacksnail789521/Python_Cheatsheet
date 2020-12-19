import re

test = "X***AA***B**@**CCC***#**Y"


for element in re.finditer(r"[^\*#@+]+", test):
    index_start, index_end = element.start(), element.end()
    print(test[index_start : index_end])