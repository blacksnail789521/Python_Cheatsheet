import pandas as pd


def get_two_indices(use_same_index = True):
    
    index_1 = pd.Index([0, 1, 2, 3], name = "index_1")
    
    if use_same_index == True:
        index_2 = index_1.rename("index_2")
    else:
        index_2 = pd.Index([-3, -2, -1, 0], name = "index_2")
    
    return index_1, index_2


def test_concat_vertically(use_same_index = True):
    
    index_1, index_2 = get_two_indices(use_same_index)
    df1 = pd.DataFrame({"first": [1, 2, 3, 4], \
                        "second": ["A", "B", "C", "D"]}, \
                       index = index_1)
    df2 = pd.DataFrame({"second": ["D", "E", "F", "G"], \
                        "first": [4, 5, 6, 7]}, \
                       index = index_2)
    print(df1)
    print(df2)
    
    # Use list to concat. (Similar to joining on column_name.)
    df_list = [df1, df2]
    df = pd.concat(df_list, ignore_index = True)
    
    # Drop duplicate rows. (Good habit.)
    df = df.drop_duplicates().reset_index(drop = True)
    
    print("-------------------------")
    print("Concatenate vertically:")
    print("-------------------------")
    print(df)


def test_concat_horizontally(use_same_index = True):
    
    index_1, index_2 = get_two_indices(use_same_index)
    df1 = pd.DataFrame({"first": [1, 2, 3, 4], \
                        "duplicate": ["A", "B", "C", "D"]}, \
                       index = index_1)
    df2 = pd.DataFrame({"second": [4, 5, 6, 7], \
                        "duplicate": ["A", "B", "C", "D"]}, \
                       index = index_2)
    print(df1)
    print(df2)
    
    # Use list to concat. (Similar to joining on index.)
    df_list = [df1, df2]
    df = pd.concat(df_list, axis = 1)
    
    # Drop duplicate column_name. (Good habit.)
    df = df.loc[ : ,~ df.columns.duplicated() ]
    
    print("-------------------------")
    print("Concatenate horizontally:")
    print("-------------------------")
    print(df)


def main():
    
    use_same_index = True
    # use_same_index = False
    
    test_concat_vertically(use_same_index)
    # test_concat_horizontally(use_same_index)


if __name__ == "__main__":
    
    main()