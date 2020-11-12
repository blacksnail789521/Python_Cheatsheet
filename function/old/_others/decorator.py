def decorator(function):
    def wrapper():
        print("decorator")

    return wrapper


@decorator
def say_hi():
    print('hello there')
    return "YA"

print(say_hi())