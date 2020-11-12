import pandas as pd
from datetime import datetime

df = pd.DataFrame({ "WAFER_ID": [3, 2, 1, 2],
                    "create_date": ["2019/09/06 09:18:38", "2019/09/06 09:17:38", \
                                    "2019/09/06 09:16:38", "2019/09/06 09:15:38"] })
for datetime_format in ["%m/%d %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f"]:
    try:
        df["create_date"] = pd.to_datetime(df["create_date"], format = datetime_format)
        successful = True
    except:
        successful = False
    if successful == True:
        break
if successful == False:
    raise Exception("WTF?")
else:
    print(df)

df = df.sort_values(by = ["WAFER_ID", "create_date"]).reset_index(drop = True)
print(df)