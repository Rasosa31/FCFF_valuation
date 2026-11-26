import requests
import pandas as pd
import math

SEC_HEADERS = {"User-Agent": "Antigravity Valuation Tool rasosa2001@gmail.com"}

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

def get_interest_expense_from_edgar(cik, ticker=None, company_name=None):
    """
    Attempts to extract gross interest expense caused or paid on interest-bearing debt from SEC EDGAR facts and 10-K filings.
    REGLA ESTRICTA: Se prohíbe de manera absoluta cualquier etiqueta o partida que contenga la palabra 'net'.
    Returns dict:
      {
         'gross_interest': float, # in millions of USD
         'source': str,
         'note': str,
         'is_net': bool
      }
    """
    gross_val = 0.0
    source = ""
    note = ""

    # 1. Búsqueda en hechos XBRL de SEC EDGAR (excluyendo cualquier etiqueta con 'net')
    if cik:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            res = requests.get(url, headers=SEC_HEADERS, timeout=10)
            if res.status_code == 200:
                us_gaap = res.json().get('facts', {}).get('us-gaap', {})
                gross_tags = [
                    'InterestExpenseDebt',
                    'InterestExpenseBorrowings',
                    'InterestAndDebtExpense',
                    'InterestCostsIncurred',
                    'InterestExpense'
                ]
                # Filtrar estrictamente: NINGUNA etiqueta puede contener 'net'
                valid_tags = [t for t in gross_tags if 'net' not in t.lower() and t in us_gaap]
                if valid_tags:
                    val = extract_latest_value(us_gaap, valid_tags)
                    if val > 0:
                        gross_val = round(val / 1_000_000, 2)
                        source = f"SEC EDGAR XBRL ({valid_tags[0]})"
                        note = f"Gasto bruto por intereses de deuda obtenido de SEC EDGAR ({valid_tags[0]}): ${gross_val:,.2f}M."
        except Exception as e:
            print(f"   ⚠️ Error en consulta XBRL de intereses: {e}")

    # 2. Si no se encontró en XBRL o arrojó 0, recurrir a los filings 10-K oficiales (Notas de Deuda)
    if gross_val <= 0.0 and ticker:
        try:
            from llm_service import get_interest_expense_from_llm
            llm_res = get_interest_expense_from_llm(ticker, company_name=company_name or "")
            if llm_res and not llm_res.get('is_net', True) and llm_res.get('gross_interest', 0.0) > 0.0:
                gross_val = llm_res['gross_interest']
                source = llm_res.get('source', "SEC Form 10-K (Nota de Deuda)")
                note = llm_res.get('note', f"Intereses brutos causados por deuda extraídos de SEC Form 10-K: ${gross_val:,.2f}M.")
        except Exception as e:
            print(f"   ⚠️ Error en consulta de filings 10-K para intereses: {e}")

    return {
        'gross_interest': gross_val,
        'source': source,
        'note': note,
        'is_net': False
    }

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


def get_options_and_unvested_data(cik, ticker=None, company_name=None):
    """
    Retrieves employee stock options (from SEC EDGAR XBRL) and unvested RSUs / PSUs
    (via SEC filings and LLM analysis of Stock-Based Compensation disclosures).
    Generates an explicit note when no options, RSUs or PSUs are disclosed.
    """
    options_data = get_options_data(cik) if cik else {
        'option_shares_millions': 0.0,
        'strike_price': 0.0,
        'option_maturity': 0.0
    }
    
    unvested_info = {'unvested_rsus': 0.0, 'unvested_psus_target': 0.0, 'note': '', 'found': False}
    if ticker:
        try:
            from llm_service import get_unvested_equity_from_llm
            unvested_info = get_unvested_equity_from_llm(ticker, company_name=company_name or "")
        except Exception as e:
            print(f"⚠️ Error al consultar unvested RSUs/PSUs: {e}")

    has_options = (options_data.get('option_shares_millions', 0.0) > 0 and options_data.get('strike_price', 0.0) > 0)
    has_rsus = (unvested_info.get('unvested_rsus', 0.0) > 0)
    has_psus = (unvested_info.get('unvested_psus_target', 0.0) > 0)
    has_any = (has_options or has_rsus or has_psus)

    if not has_any:
        unvested_note = "Nota: No se identificó información reportada sobre employee options, ni unvested RSUs, ni unvested PSUs en los reportes regulatorios (10-K/10-Q/DEF 14A) de la empresa."
    else:
        unvested_note = unvested_info.get('note', '')

    return {
        'option_shares_millions': options_data.get('option_shares_millions', 0.0),
        'strike_price': options_data.get('strike_price', 0.0),
        'option_maturity': options_data.get('option_maturity', 0.0),
        'unvested_rsus': unvested_info.get('unvested_rsus', 0.0),
        'unvested_psus_target': unvested_info.get('unvested_psus_target', 0.0),
        'has_any_equity_awards': has_any,
        'note': unvested_note
    }
