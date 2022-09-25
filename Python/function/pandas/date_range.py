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
                           freq = 'min') # same as '1min'
print(date_index)
'''
DatetimeIndex(['2022-08-25 12:00:00', '2022-08-25 12:01:00',
               '2022-08-25 12:02:00', '2022-08-25 12:03:00',
               '2022-08-25 12:04:00'],
              dtype='datetime64[ns]', freq='T')
'''


print('------------------')
date_index = pd.date_range(start = datetime(2022, 8, 25, 12, 0, 0), 
                           end = datetime(2022, 8, 25, 12, 1, 0), 
                           freq = '10S')
print(date_index)
'''
DatetimeIndex(['2022-08-25 12:00:00', '2022-08-25 12:00:10',
               '2022-08-25 12:00:20', '2022-08-25 12:00:30',
               '2022-08-25 12:00:40', '2022-08-25 12:00:50',
               '2022-08-25 12:01:00'],
              dtype='datetime64[ns]', freq='10S')
'''

