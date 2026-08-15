import pandas as pd

try:
    url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
    print(f"\n--- Testing {url} ---")
    df = pd.read_excel(url, sheet_name="Industry Averages", skiprows=8)
    print("Columns in mgnroc.xls:")
    for i, col in enumerate(df.columns):
        if 'sale' in str(col).lower() or 'cap' in str(col).lower() or 'inv' in str(col).lower():
            print(f"  --> {col}")
except Exception as e:
    print(f"Error: {e}")
