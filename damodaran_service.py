import pandas as pd
import difflib
import requests
from bs4 import BeautifulSoup
import re

def get_damodaran_erp():
    """Scrapes the live Implied ERP from Damodaran's Homepage"""
    print("   -> Contactando NYU Stern (Damodaran Implied ERP)...")
    try:
        url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        
        match = re.search(r'Implied ERP.*?=\s*(\d+\.\d+)%', text, re.IGNORECASE)
        if match:
            erp_value = float(match.group(1)) / 100
            print(f"   -> ERP Extraído: {erp_value:.4%}")
            return erp_value
    except Exception as e:
        print(f"Error scraping ERP: {e}")
    return 0.0460 # Default fallback

def get_damodaran_metrics(query_industry):
    """
    Downloads Damodaran's datasets and fuzzy-matches the yfinance industry 
    to Damodaran's classifications to extract Unlevered Beta and Sales/Capital ratio.
    """
    metrics = {
        'unlevered_beta': None,
        'sales_to_capital': None
    }
    
    try:
        # 1. Fetch Betas
        print("   -> Contactando NYU Stern (Damodaran Betas)...")
        beta_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls"
        beta_df = pd.read_excel(beta_url, sheet_name="Industry Averages", skiprows=8)
        
        ind_col = None
        for col in beta_df.columns:
            if 'Industry' in str(col) or 'Industry' in str(beta_df[col].iloc[0]):
                ind_col = col
                break
        if not ind_col: ind_col = beta_df.columns[0]
        
        damodaran_industries = beta_df[ind_col].dropna().astype(str).tolist()
        matches = difflib.get_close_matches(query_industry, damodaran_industries, n=1, cutoff=0.3)
        if matches:
            matched_ind = matches[0]
            metrics['matched_industry'] = matched_ind
            print(f"   -> Match de Industria (Beta): '{query_industry}' -> '{matched_ind}'")
            for col in beta_df.columns:
                header_name = str(beta_df[col].iloc[0]).strip()
                if header_name == 'Unlevered beta':
                    row = beta_df[beta_df[ind_col] == matched_ind]
                    if not row.empty:
                        metrics['unlevered_beta'] = float(row[col].values[0])
                elif 'Unlevered beta corrected for cash' in header_name and 'Over time' not in header_name:
                    row = beta_df[beta_df[ind_col] == matched_ind]
                    if not row.empty:
                        metrics['unlevered_beta_cash'] = float(row[col].values[0])
    except Exception as e:
        print(f"Error scraping Damodaran Betas: {e}")

    try:
        # 2. Fetch StCR
        print("   -> Contactando NYU Stern (Damodaran Sales to Capital)...")
        stcr_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
        stcr_df = pd.read_excel(stcr_url, sheet_name="Industry Averages", skiprows=8)
        stcr_ind_col = stcr_df.columns[0]
        
        damodaran_stcr_industries = stcr_df[stcr_ind_col].dropna().astype(str).tolist()
        matches = difflib.get_close_matches(query_industry, damodaran_stcr_industries, n=1, cutoff=0.3)
        if matches:
            matched_ind = matches[0]
            print(f"   -> Match de Industria (StCR): '{query_industry}' -> '{matched_ind}'")
            for col in stcr_df.columns:
                target_col = str(col).lower()
                if 'sales/capital' in target_col or 'sales to capital' in target_col or 'sales/ invested capital' in target_col:
                    row = stcr_df[stcr_df[stcr_ind_col] == matched_ind]
                    if not row.empty:
                        metrics['sales_to_capital'] = float(row[col].values[0])
                        break
    except Exception as e:
        print(f"Error scraping Damodaran StCR: {e}")
        
    return metrics
