import pandas as pd

Cars = {"Brand": ["Honda Civic", "Toyota Corolla", "Ford Focus", "Audi A4"], \
        "Price": [22000, 25000, 27000, 35000]}

df = pd.DataFrame(Cars)
print(df)

renamed_columns_dict = {}
for attribute_name in df.columns:
    renamed_columns_dict[attribute_name] = attribute_name.upper()
df.rename(columns = renamed_columns_dict, inplace = True)
print(df)