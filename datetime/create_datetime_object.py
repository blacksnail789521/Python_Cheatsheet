from datetime import datetime


'''
format: datetime(year, month, day, hour, minute, second, microsecond)

note: We must have year, month, day
'''

dt = datetime(2022, 11, 1)
print(dt) # 2022-11-01 00:00:00

dt = datetime(2022, 11, 1, 12)
print(dt) # 2022-11-01 12:00:00

dt = datetime(2022, 11, 1, 12, 34, 56)
print(dt) # 2022-11-01 12:34:56