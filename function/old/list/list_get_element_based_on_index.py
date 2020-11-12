a = ["CH_WEX_1", "CH_WEX_1", "CH_WEX_2", "PM_WEX_3"]
index = [0, 2, 3]

a  = [ a[i] for i in range(len(a)) if i in index ]
print(a)