from datetime import datetime

d1 = "2019/07/18 12:44:10.195"
d1 = datetime.strptime(d1, "%Y/%m/%d %H:%M:%S.%f")

print(d1)
print(type(d1))