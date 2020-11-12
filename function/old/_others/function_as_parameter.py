def main():
    a = 1
    print(a)
    return a

def void_function(func):
    print("First!")
    return func()
    

if __name__ == "__main__":
    everything_you_want_to_see = void_function(lambda : main())