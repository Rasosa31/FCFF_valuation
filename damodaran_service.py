import pandas as pd
import difflib
import requests
from bs4 import BeautifulSoup
import re


def get_damodaran_erp():
    """Scrapes the live Implied ERP from Damodaran's Homepage."""
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

    return 0.0460  # Default fallback


def get_damodaran_metrics(query_industry):
    """
    Downloads Damodaran's Sales-to-Capital dataset and fuzzy-matches the
    yfinance industry classification to Damodaran's industry classification.

    Important: this service no longer downloads or supplies Damodaran beta.
    Company beta is supplied separately as Levered Beta (e.g. Yahoo Finance).
    """
    metrics = {
        'sales_to_capital': None,
        'matched_industry': None
    }

    try:
        print("   -> Contactando NYU Stern (Damodaran Sales to Capital)...")
        stcr_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
        stcr_df = pd.read_excel(stcr_url, sheet_name="Industry Averages", skiprows=8)

        stcr_ind_col = stcr_df.columns[0]
        damodaran_stcr_industries = (
            stcr_df[stcr_ind_col].dropna().astype(str).tolist()
        )

        matches = difflib.get_close_matches(
            query_industry,
            damodaran_stcr_industries,
            n=1,
            cutoff=0.3
        )

        if matches:
            matched_ind = matches[0]
            metrics['matched_industry'] = matched_ind
            print(
                f"   -> Match de Industria (StCR): "
                f"'{query_industry}' -> '{matched_ind}'"
            )

            row = stcr_df[stcr_df[stcr_ind_col] == matched_ind]
            if not row.empty:
                for col in stcr_df.columns:
                    target_col = str(col).lower()
                    if (
                        'sales/capital' in target_col
                        or 'sales to capital' in target_col
                        or 'sales/ invested capital' in target_col
                    ):
                        metrics['sales_to_capital'] = float(row[col].values[0])
                        break

    except Exception as e:
        print(f"Error scraping Damodaran StCR: {e}")

    return metrics
