import pandas as pd
from datetime import datetime

# list of frequency aliases.
'''
A, Y    year end frequency
M       month end frequency
W       weekly frequency
D       calendar day frequency
H       hourly frequency
T, min  minutely frequency
S       secondly frequency
L, ms   milliseconds
U, us   microseconds
N       nanoseconds
'''

print('------------------')
date_index = pd.date_range(start = '2022-08-25 12:00:00', 
                           periods = 5, 
                           freq = 'min')
print(date_index)
'''
DatetimeIndex(['2022-08-25 12:00:00', '2022-08-25 12:01:00',
               '2022-08-25 12:02:00', '2022-08-25 12:03:00',
               '2022-08-25 12:04:00'],
              dtype='datetime64[ns]', freq='T')
'''


print('------------------')
date_index = pd.date_range(start = datetime(2022, 8, 25, 12, 00, 00), 
                           end = datetime(2022, 8, 25, 12, 00, 5), 
                           freq = 'S')
print(date_index)
'''
DatetimeIndex(['2022-08-25 12:00:00', '2022-08-25 12:00:01',
               '2022-08-25 12:00:02', '2022-08-25 12:00:03',
               '2022-08-25 12:00:04', '2022-08-25 12:00:05'],
              dtype='datetime64[ns]', freq='S')
'''

