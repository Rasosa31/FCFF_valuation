import requests
import pandas as pd
import math

SEC_HEADERS = {"User-Agent": "Antigravity Agent (rasosa2001@gmail.com)"}

def get_cik_from_ticker(ticker):
    """
    Fetches the CIK number for a given stock ticker from the SEC website.
    Returns a zero-padded 10-digit CIK string.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        res = requests.get(url, headers=SEC_HEADERS, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        target = str(ticker).strip().upper()
        for key, val in data.items():
            if val.get('ticker') == target:
                cik_int = val.get('cik_str')
                # SEC APIs expect a 10-digit zero-padded CIK
                return str(cik_int).zfill(10)
    except Exception as e:
        print(f"Error mapping ticker to CIK: {e}")
    return None

def extract_latest_value(facts, tags):
    """
    Helper to search a list of tags in the SEC facts dictionary and return the most recent 10-K value.
    """
    for tag in tags:
        concept = facts.get(tag)
        if concept and 'units' in concept:
            # Grab the first available unit key (e.g., 'shares', 'USD', 'pure')
            units_key = list(concept['units'].keys())[0]
            try:
                df = pd.DataFrame(concept['units'][units_key])
                df = df[df['form'] == '10-K'].copy()
                if not df.empty:
                    df['end'] = pd.to_datetime(df['end'])
                    df = df.sort_values(by='end')
                    # Get the most recent value
                    return float(df.iloc[-1]['val'])
            except Exception:
                pass
    return 0.0

def get_options_data(cik):
    """
    Retrieves outstanding options, weighted average exercise price, and average maturity.
    """
    result = {
        'option_shares_millions': 0.0,
        'strike_price': 0.0,
        'option_maturity': 0.0
    }
    
    if not cik:
        return result
        
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        res = requests.get(url, headers=SEC_HEADERS, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        us_gaap = data.get('facts', {}).get('us-gaap', {})
        if not us_gaap:
            return result
            
        # 1. Outstanding Options
        options_tags = [
            'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber',
            'ShareBasedCompensationArrangementsByShareBasedPaymentAwardOptionsOutstandingNumber'
        ]
        raw_options = extract_latest_value(us_gaap, options_tags)
        # Convert to millions to match our app standard
        result['option_shares_millions'] = round(raw_options / 1_000_000, 2)
        
        # 2. Strike Price
        strike_tags = [
            'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice'
        ]
        result['strike_price'] = round(extract_latest_value(us_gaap, strike_tags), 2)
        
        # 3. Option Maturity (Remaining Contractual Term)
        # In SEC API, duration/maturity is often reported as string like P4Y2M10D or as a pure decimal.
        maturity_tags = [
            'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageRemainingContractualTerm2'
        ]
        
        # We need a custom extractor for maturity because its 'val' might be a float or a timedelta string
        concept = us_gaap.get(maturity_tags[0])
        if concept and 'units' in concept:
            units_key = list(concept['units'].keys())[0]
            df = pd.DataFrame(concept['units'][units_key])
            df = df[df['form'] == '10-K']
            if not df.empty:
                df['end'] = pd.to_datetime(df['end'])
                df = df.sort_values(by='end')
                raw_maturity = df.iloc[-1]['val']
                # Sometimes it's a number (years), sometimes it's a string ISO period
                if isinstance(raw_maturity, (int, float)):
                    result['option_maturity'] = float(raw_maturity)
                elif isinstance(raw_maturity, str):
                    # SEC ISO durations like P4Y3M -> 4 years, 3 months
                    years = 0.0
                    try:
                        if 'Y' in raw_maturity:
                            # basic extraction
                            y_str = raw_maturity.split('P')[-1].split('Y')[0]
                            years += float(y_str)
                            if 'M' in raw_maturity:
                                m_str = raw_maturity.split('Y')[-1].split('M')[0]
                                years += float(m_str) / 12.0
                        result['option_maturity'] = round(years, 2)
                    except Exception:
                        result['option_maturity'] = 5.0 # default fallback
        
    except Exception as e:
        print(f"Error fetching SEC Company Facts: {e}")
        
    return result
