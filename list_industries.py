import pandas as pd
import json
url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
df = pd.read_excel(url, sheet_name="Industry Averages", skiprows=8)
stcr_ind_col = df.columns[0]
industries = df[stcr_ind_col].dropna().astype(str).tolist()
with open("damodaran_industries.json", "w") as f:
    json.dump(industries, f)
print("Industries saved.")
