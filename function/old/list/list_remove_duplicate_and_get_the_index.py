from collections import OrderedDict

a = ["CH_WEX_1", "CH_WEX_1", "CH_WEX_2", "PM_WEX_3"]

d = OrderedDict((element, a.index(element)) for element in a)
print( list(d.keys()) )
print( list(d.values()) )