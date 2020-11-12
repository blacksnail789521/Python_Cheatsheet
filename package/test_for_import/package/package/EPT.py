import os

def EPT_DATA(a):
    a = a + 1
    print("a:", a)
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "test.txt"), "r") as file:
        for line in file:
            print(line)