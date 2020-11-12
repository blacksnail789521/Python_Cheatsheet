from collections import OrderedDict

merge_waiting_dict = OrderedDict()

merge_waiting_dict["1"] = {"length": 5, "count": 10}
merge_waiting_dict["2"] = {"length": 3, "count": 10}
merge_waiting_dict["3"] = {"length": 4, "count": 10}
merge_waiting_dict["4"] = {"length": 4, "count": 8}

print(merge_waiting_dict)

# Sort by length, count.
merge_waiting_dict = OrderedDict( sorted(merge_waiting_dict.items(), key = lambda item : (item[1]["length"], item[1]["count"])) )
print(merge_waiting_dict)