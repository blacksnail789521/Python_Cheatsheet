import pandas as pd

ept_data = pd.DataFrame({ "WAFER_ID": [1, 5, 7, 5],
                     "value": ["foo", "bar", "baz", "foo"] })
defect_data = pd.DataFrame({ "WAFER_ID": [5, 6, 7, 8],
                     "value": ["foo", "bar", "baz", "foo"] })

print(ept_data)
print(defect_data)

#ept_data = pd.merge(ept_data, defect_data[["WAFER_ID"]], on = "WAFER_ID", how = "inner")
#print(ept_data)

# Faster version.
ept_data = ept_data.loc[ept_data["WAFER_ID"].isin( defect_data["WAFER_ID"] )].reset_index(drop = True)
print(ept_data)