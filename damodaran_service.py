import pandas as pd
import difflib
import requests
from bs4 import BeautifulSoup
import re
import os


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
    industry classification to Damodaran's industry classification.

    Important: this service no longer downloads or supplies Damodaran beta.
    Company beta is supplied separately as Levered Beta (e.g. Yahoo Finance).
    """
    metrics = {
        'sales_to_capital': None,
        'matched_industry': None
    }

    if not query_industry:
        return metrics

    try:
        print("   -> Contactando NYU Stern (Damodaran Sales to Capital)...")
        stcr_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
        stcr_df = pd.read_excel(stcr_url, sheet_name="Industry Averages", skiprows=8)

        stcr_ind_col = stcr_df.columns[0]
        damodaran_stcr_industries = (
            stcr_df[stcr_ind_col].dropna().astype(str).tolist()
        )

        # Primero intenta match exacto (case-insensitive)
        exact = [ind for ind in damodaran_stcr_industries if ind.lower() == str(query_industry).lower()]
        if exact:
            matched_ind = exact[0]
        else:
            # Fuzzy match con cutoff más estricto
            matches = difflib.get_close_matches(
                str(query_industry),
                damodaran_stcr_industries,
                n=1,
                cutoff=0.55
            )
            matched_ind = matches[0] if matches else None

        if matched_ind:
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
                        try:
                            metrics['sales_to_capital'] = float(row[col].values[0])
                        except (ValueError, TypeError):
                            pass
                        break
        else:
            print(f"   ⚠️ No se encontró match de StCR para: '{query_industry}'")

    except Exception as e:
        print(f"Error scraping Damodaran StCR: {e}")

    return metrics


def get_damodaran_industries_list():
    """Returns the list of Damodaran industries."""
    try:
        stcr_url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/mgnroc.xls"
        stcr_df = pd.read_excel(stcr_url, sheet_name="Industry Averages", skiprows=8)
        stcr_ind_col = stcr_df.columns[0]
        return stcr_df[stcr_ind_col].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error fetching Damodaran industries: {e}")
        return []


def get_damodaran_industry_from_indname(ticker, company_name=None):
    """
    Return the Damodaran *Industry Group* for a given ticker or company name.

    Estrategias:
    1. Match por ticker. Si hay varios, usa el nombre de la compañía para desambiguar.
       Si no hay nombre, prioriza bolsas de EE.UU. (NYSE, NASDAQ, etc.).
    2. Match exacto por nombre de empresa.
    3. Substring por nombre.
    4. Fuzzy match por nombre.
    """
    file_path = "indname_cache.csv"
    if not os.path.exists(file_path):
        print("   ⚠️ indname_cache.csv no encontrado")
        return None

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"   -> indname_cache cargado: {len(df)} filas")

        def _normalize(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r"[\.,&;:\(\)\[\]\{\}\"']", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        ticker_str = str(ticker).upper().strip()

        # ---------------------------------------------------------------
        # 1. Match por ticker + desambiguación por nombre / bolsa US
        # ---------------------------------------------------------------
        if 'Exchange:Ticker' in df.columns:
            exact_match = df[
                df['Exchange:Ticker']
                .astype(str)
                .str.upper()
                .str.endswith(f":{ticker_str}", na=False)
            ].copy()

            if not exact_match.empty:
                # Si el usuario dio nombre de compañía, usarlo para elegir la mejor fila
                if company_name and 'Company Name' in exact_match.columns:
                    clean_input = _normalize(company_name)
                    exact_match = exact_match.copy()
                    exact_match['_norm_name'] = exact_match['Company Name'].astype(str).apply(_normalize)

                    # 1a. Coincidencia exacta de nombre normalizado
                    name_exact = exact_match[exact_match['_norm_name'] == clean_input]
                    if not name_exact.empty:
                        row = name_exact.iloc[0]
                        industry = row['Industry Group']
                        print(f"   ✅ Match ticker + nombre exacto: {row['Exchange:Ticker']} → {industry}")
                        return industry

                    # 1b. El nombre del usuario aparece dentro del Company Name
                    if len(clean_input) >= 4:
                        name_contains = exact_match[
                            exact_match['_norm_name'].str.contains(clean_input, na=False, regex=False)
                        ]
                        if not name_contains.empty:
                            row = name_contains.iloc[0]
                            industry = row['Industry Group']
                            print(f"   ✅ Match ticker + nombre (contains): {row['Exchange:Ticker']} → {industry}")
                            return industry

                    # 1c. Fuzzy sobre los candidatos del mismo ticker
                    candidates = exact_match['Company Name'].dropna().astype(str).tolist()
                    fuzzy = difflib.get_close_matches(company_name, candidates, n=1, cutoff=0.6)
                    if fuzzy:
                        row = exact_match[exact_match['Company Name'] == fuzzy[0]].iloc[0]
                        industry = row['Industry Group']
                        print(f"   ✅ Match ticker + nombre fuzzy: {row['Exchange:Ticker']} ('{fuzzy[0]}') → {industry}")
                        return industry

                # Si no se pudo desambiguar por nombre → priorizar bolsas de EE.UU.
                us_exchanges = ('NYSE:', 'NASDAQ:', 'AMEX:', 'NYSEARCA:', 'BATS:')
                us_match = exact_match[
                    exact_match['Exchange:Ticker']
                    .astype(str)
                    .str.upper()
                    .str.startswith(us_exchanges, na=False)
                ]

                if not us_match.empty:
                    row = us_match.iloc[0]
                    industry = row['Industry Group']
                    print(f"   ✅ Match exacto por ticker (US): {row['Exchange:Ticker']} → {industry}")
                    return industry
                else:
                    row = exact_match.iloc[0]
                    industry = row['Industry Group']
                    print(f"   ✅ Match exacto por ticker: {row['Exchange:Ticker']} → {industry}")
                    return industry

        # ---------------------------------------------------------------
        # 2-4. Name-based matching (cuando no hubo match de ticker)
        # ---------------------------------------------------------------
        if company_name and 'Company Name' in df.columns:
            clean_input = _normalize(company_name)
            if not clean_input:
                print(f"   ❌ Nombre de empresa vacío después de normalizar")
                return None

            normalized_names = df['Company Name'].astype(str).apply(_normalize)

            # 2. Exact name match
            exact_name_match = df[normalized_names == clean_input]
            if not exact_name_match.empty:
                industry = exact_name_match.iloc[0]['Industry Group']
                print(f"   ✅ Match exacto por nombre: '{company_name}' → {industry}")
                return industry

            # 3. Substring match
            if len(clean_input) >= 6:
                substring_match = df[
                    normalized_names.str.contains(clean_input, na=False, regex=False)
                ]
                if not substring_match.empty:
                    industry = substring_match.iloc[0]['Industry Group']
                    print(f"   ✅ Match por substring: '{company_name}' → {industry}")
                    return industry

            # 4. Fuzzy match
            raw_names = df['Company Name'].dropna().astype(str).tolist()
            fuzzy = difflib.get_close_matches(company_name, raw_names, n=1, cutoff=0.75)
            if fuzzy:
                matched_row = df[df['Company Name'] == fuzzy[0]]
                if not matched_row.empty:
                    industry = matched_row.iloc[0]['Industry Group']
                    print(f"   ✅ Match fuzzy: '{company_name}' → '{fuzzy[0]}' → {industry}")
                    return industry

        print(f"   ❌ No se encontró industria Damodaran para ticker={ticker_str}, name={company_name}")
        return None

    except Exception as e:
        print(f"Error reading indname cache: {e}")
        return None