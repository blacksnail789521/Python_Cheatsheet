from collections import OrderedDict
from pprint import pprint


KPI_dict = OrderedDict()
KPI_dict["factor_1"] = {"TP": 0.7, "recall": 0.7, "final_score": 0.7}
KPI_dict["factor_2"] = {"TP": 0.4, "recall": 0.6, "final_score": 0.5}
KPI_dict["factor_3"] = {"TP": 0.6, "recall": 0.4, "final_score": 0.5}

print("----------------------------")
print("before:")
pprint(KPI_dict)

KPI_dict = OrderedDict( sorted(KPI_dict.items(), key = lambda item : (item[1]["final_score"], item[1]["TP"]), reverse = True) )

print("----------------------------")
print("after:")
pprint(KPI_dict)