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

def get_damodaran_industries_list():
    """Returns the list of 96 Damodaran industries."""
    try:
        stcr_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
        stcr_df = pd.read_excel(stcr_url, sheet_name="Industry Averages", skiprows=8)
        stcr_ind_col = stcr_df.columns[0]
        return stcr_df[stcr_ind_col].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error fetching Damodaran industries: {e}")
        return []

def get_damodaran_industry_from_indname(ticker, company_name=None):
    """Return the Damodaran *Industry Group* for a given ticker or company name.

    The function reads the cached ``indname_cache.csv`` (generated from
    ``indname.xls``).  It attempts the following strategies in order:

    1. **Exact ticker match** – looks for rows where the ``Exchange:Ticker``
       column ends with ``:{ticker}`` (case‑insensitive).  Some files include
       the exchange prefix (e.g. ``NYSE:MRK``).  If a direct match is found the
       corresponding ``Industry Group`` is returned.
    2. **Exact name match** – normalises the supplied ``company_name`` and the
       ``Company Name`` column (lower‑case, stripped punctuation, collapsed
       whitespace) and checks for equality.
    3. **Substring name match** – after normalisation, checks whether the
       cleaned ``company_name`` appears as a substring of any ``Company Name``
       entry.
    4. **Fuzzy name match** – uses ``difflib.get_close_matches`` on the raw
       names as a last resort (cutoff 0.7).

    If none of the strategies succeed the function returns ``None`` so that
    callers can fall back to Yahoo Finance industry information.
    """
    import os
    import difflib
    import re

    file_path = "indname_cache.csv"
    if not os.path.exists(file_path):
        # Cache missing – callers will handle the fallback.
        return None

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # -----------------------------------------------------------------
        # Helper: normalise a string for robust comparison.
        # -----------------------------------------------------------------
        def _normalize(text: str) -> str:
            text = text.lower()
            # Remove punctuation (.,&;:() etc.) and extra whitespace.
            #text = re.sub(r"[\.,&;:\(\)\[\]\{\}"'"]", " ", text)
            text = re.sub(r"[\.,&;:\(\)\[\]\{\}\"']", " ", text)              
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # ---------------------------------------------------------------
        # 1. Exact ticker match – also accept any occurrence of the ticker.
        # ---------------------------------------------------------------
        if 'Exchange:Ticker' in df.columns:
            # Ensure ticker is a string and uppercase for comparison.
            ticker_str = str(ticker).upper()
            # Direct ``endswith`` match (e.g. ``NYSE:MRK``)
            exact_match = df[df['Exchange:Ticker'].astype(str).str.upper().str.endswith(f":{ticker_str}", na=False)]
            if not exact_match.empty:
                return exact_match.iloc[0]['Industry Group']
            # Fallback: ticker appears anywhere in the column.
            broader_match = df[df['Exchange:Ticker'].astype(str).str.upper().str.contains(ticker_str, na=False)]
            if not broader_match.empty:
                return broader_match.iloc[0]['Industry Group']

        # ---------------------------------------------------------------
        # 2‑4. Name based matching (requires a supplied company name).
        # ---------------------------------------------------------------
        if company_name and 'Company Name' in df.columns:
            clean_input = _normalize(company_name)

            # 2. Exact normalized name equality.
            normalized_names = df['Company Name'].astype(str).apply(_normalize)
            exact_name_match = df[normalized_names == clean_input]
            if not exact_name_match.empty:
                return exact_name_match.iloc[0]['Industry Group']

            # 3. Substring match on normalized strings.
            substring_match = df[normalized_names.str.contains(clean_input, na=False)]
            if not substring_match.empty:
                return substring_match.iloc[0]['Industry Group']

            # 4. Fuzzy match on the raw names (keeps original spelling).
            raw_names = df['Company Name'].dropna().astype(str).tolist()
            fuzzy = difflib.get_close_matches(company_name, raw_names, n=1, cutoff=0.7)
            if fuzzy:
                matched_row = df[df['Company Name'] == fuzzy[0]]
                if not matched_row.empty:
                    return matched_row.iloc[0]['Industry Group']

        # No match found.
        return None
    except Exception as e:
        print(f"Error reading indname cache: {e}")
        return None

