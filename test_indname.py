import pandas as pd
url = "https://www.stern.nyu.edu/~adamodar/pc/datasets/indname.xls"
try:
    df = pd.read_excel(url, sheet_name=None)
    for name, sheet in df.items():
        print(f"Sheet: {name}, columns: {list(sheet.columns)}")
        print(sheet.head(5))
except Exception as e:
    print("Error:", e)
