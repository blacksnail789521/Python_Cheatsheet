import pandas as pd
import numpy as np

#single_recipe_table = pd.DataFrame({ "RECIPE_CNT": [4, 4, 4, 4],
#                    "REMARK": ["X", "X", "X", np.nan] })


single_recipe_table = pd.DataFrame()

single_recipe_table["RECIPE_CNT"] = [4, 4, 4, 4]
single_recipe_table["REMARK"] = [np.nan] * len(single_recipe_table)

single_recipe_table.loc[0, "REMARK"] = "X"

print(single_recipe_table)


RECIPE_CNT = single_recipe_table["RECIPE_CNT"][0]
#REMARK_CNT = len( single_recipe_table.loc[ single_recipe_table["REMARK"] != np.nan ] )
REMARK_CNT = single_recipe_table["REMARK"].isnull().sum()
print(RECIPE_CNT, REMARK_CNT)


a = {}

a.update({"XD": 1, "HA": 2})
a.update(a = 3, b = 4)
print(a)