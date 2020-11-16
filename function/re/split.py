import re


enclosure_punctuation = "parentheses"
# enclosure_punctuation = "brackets"

test_string = "X***AA***B**@**CCC***#**Y**+**Z"

split_list = ["*", "@", "#", "+"]

if enclosure_punctuation == "parentheses":
    # Need to use non-capturing version.
    split_string = "(?:" + "|".join(map(re.escape, split_list)) + ")+"
elif enclosure_punctuation == "brackets":
    split_string = "[" + "".join(map(re.escape, split_list)) + "]+"

split_test_string = re.split(split_string, test_string)
print(split_test_string)