import tkinter as tk
from tkinter.simpledialog import askstring


def get_input_from_input_box():
    
    root = tk.Tk()
    root.withdraw()
    user_input = askstring("title", "Enter a string:")
    
    return user_input


user_input = get_input_from_input_box()
print(user_input)