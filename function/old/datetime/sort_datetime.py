from datetime import datetime

d1 = "2019/07/18 12:44:10.195"
d1 = datetime.strptime(d1, "%Y/%m/%d %H:%M:%S.%f")
d2 = "2019/07/20 12:44:10.195"
d2 = datetime.strptime(d2, "%Y/%m/%d %H:%M:%S.%f")
d3 = "2019/07/19 12:44:10.195"
d3 = datetime.strptime(d3, "%Y/%m/%d %H:%M:%S.%f")


event_create_dt_list = [d1, d2, d3]
print(event_create_dt_list)
#event_create_dt_list = sorted( event_create_dt_list )
print(event_create_dt_list)
sorted_event_order = [sorted(event_create_dt_list).index(x) for x in event_create_dt_list]
print(sorted_event_order)