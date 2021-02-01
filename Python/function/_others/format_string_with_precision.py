# float
a = 0.123456789
print(f"{a:.2f}")

# int
a = 10
print(f"{a:5}")

# string
print(f"{'left aligned':<30}")
print(f"{'right aligned':>30}")
print(f"{'centered':^30}")
print(f"{'centered':*^30}") # Use "*" as a fill char.

# two columns (money format)
saving = 967530
stock = 44690106
interest = stock * 1.1

float_precision = 0
left_align = 10
right_align = 10
print(f"{'saving:':<{left_align}}{saving:>{right_align},.{float_precision}f}")
print(f"{'stock:':<{left_align}}{stock:>{right_align},.{float_precision}f}")
print(f"{'interest:':<{left_align}}{interest:>{right_align},.{float_precision}f}")