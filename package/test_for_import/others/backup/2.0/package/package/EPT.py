def EPT_DATA(a):
    a = a + 1
    print("a:", a)
    
    with open('test.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            print(line)