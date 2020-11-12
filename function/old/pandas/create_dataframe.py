import pandas as pd

df = pd.DataFrame({"Brand": ["Honda Civic", "Toyota Corolla", "Ford Focus", "Audi A4"], \
                   "Price": [22000, 25000, 27000, 35000]})

print(df)
print(df["Brand"].iloc[0])
print(df.iloc[0]["Brand"])