import pandas as pd
import os
from datetime import datetime


df = pd.read_csv(os.path.join('df', 'df_with_time.csv'), index_col = 'time')
df.index = pd.to_datetime(df.index)

print(df.index)
'''
DatetimeIndex(['2022-08-25 12:00:00', '2022-08-25 12:00:10',
               '2022-08-25 12:00:20', '2022-08-25 12:00:30',
               '2022-08-25 12:00:40', '2022-08-25 12:00:50',
               '2022-08-25 12:01:00'],
              dtype='datetime64[ns]', name='time', freq=None)
'''

print(df)
'''
                     A  B
time                     
2022-08-25 12:00:00  1  2
2022-08-25 12:00:10  1  2
2022-08-25 12:00:20  1  2
2022-08-25 12:00:30  1  2
2022-08-25 12:00:40  1  2
2022-08-25 12:00:50  1  2
2022-08-25 12:01:00  1  2
'''


print(df.loc[ datetime(2022, 8, 25, 12, 0, 20) : datetime(2022, 8, 25, 12, 0, 45) ])
'''
                     A  B
time                     
2022-08-25 12:00:20  1  2
2022-08-25 12:00:30  1  2
2022-08-25 12:00:40  1  2
'''