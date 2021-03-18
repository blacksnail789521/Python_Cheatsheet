import numpy as np

        
# Check for equality
np.testing.assert_allclose(sum([0.7, 0.2, 0.1]), 1)

# Check if number is an int (precision is zero) (remainder of 1 is zero)
number = np.log10(10)
print("number:", number)
print("remainder of number:", number % 1)
np.testing.assert_allclose(number % 1, 0)