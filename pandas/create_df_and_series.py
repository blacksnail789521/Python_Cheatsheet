import pandas as pd


'''
Create a df:
    
   A  B  C
0  1  2  3
1  4  5  6
2  7  8  9

'''

# dict of list.
print("------------------")
df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
print(df)

# 2d_array and factor_name_list.
print("------------------")
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns = ["A", "B", "C"])
print(df)

# list of dict.
print("------------------")
df = pd.DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 4, "B": 5, "C": 6}, {"A": 7, "B": 8, "C": 9}])
print(df)


'''
Create a series:

time
2022-11-01 12:00:00    2
2022-11-01 12:01:00    1
2022-11-01 12:02:00    0
2022-11-01 12:03:00    1
2022-11-01 12:04:00    2
Freq: T, Name: value, dtype: int64

'''
date_index = pd.date_range(start = '2022-11-01 12:00:00',
                           periods = 5, freq = 'min', name = 'time')
series = pd.Series([2, 1, 0, 1, 2], index = date_index, name = 'value')
print(series)