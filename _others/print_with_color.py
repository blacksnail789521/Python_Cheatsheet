from colorama import Fore, Back, Style


print("Hello World!")
print(Fore.RED + "some red text" + Style.RESET_ALL)
print(Back.GREEN + "and with a green background" + Style.RESET_ALL)
print(Style.DIM + "and in dim text" + Style.RESET_ALL)
print(Fore.BLUE + Back.YELLOW + "and with a green background" + Style.RESET_ALL)
