import re

attribute = "FDC_GMP_N5-5FFPO3FRCV08Z-PM1-test Mean(W11(W9(PM1:The current process gas_26 flow)))"
# Delimiters: " ", "_", "-"
split_attribute = re.split("([\s_\-])", attribute)
print(split_attribute)

split_attribute_delimiter = [ element for element in split_attribute \
                              if element == " " or element == "_" or element == "-" ]
print(split_attribute_delimiter)

split_attribute = [ element for element in split_attribute \
                    if element not in split_attribute_delimiter ]
print(split_attribute)