# float
a = 10.123456789
print(f"{a:.3f}")  # show 3 decimal places
print(f"{a:.3e}")  # show 3 decimal places in scientific notation
print(f"{a:.3g}")  # show 3 significant figures
"""
10.12
1.01e+01
10
"""

a = 0.0000012345
print(f"{a:.2f}")
print(f"{a:.2e}")
print(f"{a:.2g}")
"""
0.00
1.23e-06
1.2e-06
"""

a = 1234.5678
print(f"{a:,}")  # add comma
print(f"{a:_}")  # add underscore
"""
1,234.5678
1_234.5678
"""


# string
print(f"{'left aligned':<30}")  # left aligned in total 30 characters
print(f"{'right aligned':>30}")  # right aligned in total 30 characters
print(f"{'centered':^30}")  # centered in total 30 characters
print(f"{'centered':*^30}")  # centered in total 30 characters with * as padding


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
