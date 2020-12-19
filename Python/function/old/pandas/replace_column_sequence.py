import pandas as pd

data = pd.DataFrame({ "WAFER_ID": [1, 5, 7, 5],
                      "value": ["foo", "bar", "baz", "foo"] })
print(data)

sequence = [5, 6, 7, 8]
data["WAFER_ID"] = sequence
print(data)
