import re


test_string = "X***AA***B**@**CCC***#**Y**+**Z"

replace_from_list = ["*", "@", "#", "+"]
replace_to = "~"

replace_from_string = "|".join(map(re.escape, replace_from_list))
# replace_from_string = "|".join(replace_from_list) # Sometimes we don't want to escape.
result = re.sub(replace_from_string, replace_to, test_string)
print(result)