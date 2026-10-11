import pandas as pd
import os
import difflib

def lookup_industry(ticker, company_name=None):
    file_path = "indname_cached.xls"
    if not os.path.exists(file_path):
        print("Downloading indname.xls...")
        url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/indname.xls"
        df = pd.read_excel(url, sheet_name="By company name")
        df.to_excel(file_path, index=False)
    else:
        df = pd.read_excel(file_path)
    
    # Clean up the ticker for matching. YF tickers like 'AAPL' usually match ':AAPL' in Exchange:Ticker
    # E.g., 'NasdaqGS:AAPL'
    
    exact_match = df[df['Exchange:Ticker'].astype(str).str.endswith(f":{ticker}")]
    if not exact_match.empty:
        return exact_match.iloc[0]['Industry Group']
        
    if company_name:
        # Fuzzy match company name
        names = df['Company Name'].dropna().astype(str).tolist()
        matches = difflib.get_close_matches(company_name, names, n=1, cutoff=0.6)
        if matches:
            matched_row = df[df['Company Name'] == matches[0]]
            return matched_row.iloc[0]['Industry Group']
            
    return None

print("AAPL:", lookup_industry("AAPL", "Apple Inc."))
print("FLWS:", lookup_industry("FLWS", "1-800-FLOWERS.COM"))
print("BINHO:", lookup_industry("BINHO", "1000 Yatirimlar Holding"))
