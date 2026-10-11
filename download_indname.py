import pandas as pd
import os

print("Downloading indname.xls...")
url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/indname.xls"
df = pd.read_excel(url, sheet_name="By company name")
df.to_csv("indname_cache.csv", index=False)
print("Saved to indname_cache.csv")
