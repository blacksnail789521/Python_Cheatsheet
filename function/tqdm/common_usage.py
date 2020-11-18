import time
import pandas as pd

from tqdm import tqdm, trange
# Prevent to make logging colorless in Windows.
# import colorama
# colorama.deinit()

desc_format = " └─────🡢 "


def test_list():
    
    a = [0, 1, 2, 3, 4]
    print("list")
    for element in tqdm(a, desc = desc_format):
        time.sleep(0.25)


def test_enumerate():
    
    a = [0, 1, 2, 3, 4]
    print("enumerate")
    for index, element in enumerate(tqdm(a, desc = desc_format)):
        time.sleep(0.25)


def test_dict():
    
    enable_enumerate = False
    # enable_enumerate = True
    
    a = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    print("dict")
    if enable_enumerate == False:
        for key, value in tqdm(a.items(), desc = desc_format):
            time.sleep(0.25)
    elif enable_enumerate == True:
        for index, (key, value) in enumerate(tqdm(a.items(), desc = desc_format)):
            time.sleep(0.25)


def test_range():
    
    length = 5
    print("range")
    for index in trange(length, desc = desc_format):
        time.sleep(0.25)


def test_iterrows():
    
    df = pd.DataFrame({"A": [0, 1, 2, 3, 4], \
                       "B": [0, 1, 2, 3, 4]})
    print("df.iterrows()")
    for index, row in tqdm(df.iterrows(), total = len(df), desc = desc_format):
        time.sleep(0.25)


def main():
    
    test_list()
    # test_enumerate()
    # test_dict()
    # test_range()
    # test_iterrows()


if __name__ == "__main__":
    
    main()