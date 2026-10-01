import yfinance as yf
import pandas as pd
import numpy as np
import time
from curl_cffi import requests

def get_financial_data(ticker_symbol, target_currency=None, max_retries=3):
    """
    Fetches the necessary financial data from Yahoo Finance for a given ticker.
    Returns a dictionary with raw metrics needed for FCFF valuation.
    If target_currency is specified (e.g. 'USD'), it automatically translates
    financials and share prices to the target currency.
    
    Mejoras:
    - Usa sesión curl_cffi con impersonate="chrome" para evitar 401/crumb
    - Reintentos con backoff exponencial
    - Manejo de errores más robusto
    - Usa TTM (Last Twelve Months) para Income Statement
    - Usa Balance Sheet trimestral más reciente
    """
    
    # Crear sesión que imita Chrome (muy efectivo contra bloqueos de Yahoo)
    session = requests.Session(impersonate="chrome")
    
    def create_ticker(symbol):
        return yf.Ticker(symbol, session=session)
    
    for attempt in range(max_retries):
        try:
            ticker = create_ticker(ticker_symbol)
            
            # Attempt to get full info
            info = ticker.info
            
            try:
                # === INCOME STATEMENT → TTM (Last Twelve Months) ===
                financials = ticker.ttm_income_stmt
                
                # Fallback si ttm no está disponible
                if financials is None or financials.empty:
                    print("⚠️ ttm_income_stmt no disponible, usando financials anuales...")
                    financials = ticker.financials
                
                # === BALANCE SHEET → Trimestral más reciente ===
                balance_sheet = ticker.quarterly_balance_sheet
                
                # Fallback si trimestral no está disponible
                if balance_sheet is None or balance_sheet.empty:
                    print("⚠️ quarterly_balance_sheet no disponible, usando balance_sheet anual...")
                    balance_sheet = ticker.balance_sheet
                
                # Datos anuales (se usan para R&D histórico y margen promedio)
                annual_financials = ticker.financials
                
            except Exception as e:
                print(f"Error fetching fundamental data (intento {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None
                
            if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
                print("Financial data not available for this ticker.")
                return None

            # Columna más reciente del Income Statement (ahora es TTM)
            latest_date = financials.columns[0]
            print(f"📅 Usando Income Statement de: {latest_date} (TTM)")
            
            # Columna más reciente del Balance Sheet
            latest_bs_date = balance_sheet.columns[0]
            print(f"📅 Usando Balance Sheet de: {latest_bs_date}")
            
            def get_value(df, row_name, date_col, default=0.0):
                try:
                    val = df.loc[row_name, date_col]
                    return float(val) if pd.notna(val) else default
                except (KeyError, TypeError):
                    return default

            # 1. Income Statement Data (desde TTM)
            revenue = get_value(financials, 'Total Revenue', latest_date)
            operating_income = get_value(financials, 'Operating Income', latest_date)
            interest_expense = get_value(financials, 'Interest Expense', latest_date, default=0.0)
            # yfinance sometimes records interest expense as negative or positive, ensure positive
            interest_expense = abs(interest_expense)
            
            # R&D Expenses del período TTM
            rd_expenses_base = get_value(financials, 'Research And Development', latest_date, default=0.0)
            
            # R&D de años anteriores → se toman de los datos ANUALES
            def get_past_value_annual(df, row_name, years_back, default=0.0):
                try:
                    if df is not None and not df.empty and len(df.columns) > years_back:
                        val = df.loc[row_name, df.columns[years_back]]
                        return float(val) if pd.notna(val) else default
                except (KeyError, TypeError):
                    pass
                return default

            rd_m1 = get_past_value_annual(annual_financials, 'Research And Development', 1, 0.0)
            rd_m2 = get_past_value_annual(annual_financials, 'Research And Development', 2, 0.0)
            rd_m3 = get_past_value_annual(annual_financials, 'Research And Development', 3, 0.0)
            
            # Tax Rate Calculation: Income Tax Expense / Pretax Income (desde TTM)
            pretax_income = get_value(financials, 'Pretax Income', latest_date)
            tax_provision = get_value(financials, 'Tax Provision', latest_date)
            marginal_tax_rate = tax_provision / pretax_income if pretax_income > 0 else 0.21  # Default 21%
            marginal_tax_rate = max(0.0, min(marginal_tax_rate, 0.40))  # Cap between 0% and 40%

            # 2. Balance Sheet Data (desde el trimestral más reciente)
            cash = get_value(balance_sheet, 'Cash And Cash Equivalents', latest_bs_date)
            short_term_investments = get_value(balance_sheet, 'Other Short Term Investments', latest_bs_date, 0.0)
            total_cash = cash + short_term_investments
            
            equity = get_value(balance_sheet, 'Stockholders Equity', latest_bs_date)
            
            total_debt = get_value(balance_sheet, 'Total Debt', latest_bs_date)
            minority_interest = get_value(balance_sheet, 'Minority Interest', latest_bs_date, default=0.0)

            # 3. Market Data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
            shares_outstanding = info.get('impliedSharesOutstanding') or info.get('sharesOutstanding') or 0.0
            beta = info.get('beta', 1.0)

            # Scale factor
            scale_factor = 1_000_000
            
            # 4. Meta Information
            industry = info.get('industry', 'Unknown')
            country = info.get('country', 'Unknown')
            currency = info.get('currency', 'USD')
            financial_currency = info.get('financialCurrency', currency)
            
            # 5. Currency Translation Logic
            exchange_rate_financials = 1.0
            exchange_rate_price = 1.0
            
            if target_currency:
                if financial_currency != target_currency:
                    try:
                        print(f"💱 Solicitando Tasa de Cambio para Estados Financieros: {financial_currency} a {target_currency}...")
                        fx_ticker = create_ticker(f"{financial_currency}{target_currency}=X")
                        fetched_rate = (
                            fx_ticker.info.get('regularMarketPreviousClose') 
                            or getattr(fx_ticker.fast_info, 'previous_close', None)
                        )
                        if fetched_rate: 
                            exchange_rate_financials = fetched_rate
                            print(f"   -> Tasa aplicada: {exchange_rate_financials}")
                    except Exception as e:
                        print(f"   -> No se pudo obtener la tasa de cambio {financial_currency} a {target_currency}: {e}")
                        
                if currency != target_currency:
                    try:
                        print(f"💱 Solicitando Tasa de Cambio para Precio de Acción: {currency} a {target_currency}...")
                        fx_ticker = create_ticker(f"{currency}{target_currency}=X")
                        fetched_rate = (
                            fx_ticker.info.get('regularMarketPreviousClose') 
                            or getattr(fx_ticker.fast_info, 'previous_close', None)
                        )
                        if fetched_rate: 
                            exchange_rate_price = fetched_rate
                            print(f"   -> Tasa aplicada: {exchange_rate_price}")
                    except Exception as e:
                        print(f"   -> No se pudo obtener la tasa de cambio {currency} a {target_currency}: {e}")
            
            # Analyst Growth Estimates (from info)
            analyst_revenue_growth = info.get('revenueGrowth', 0.0)
            
            # Historical Stock Volatility Calculation (Annualized over 252 days)
            try:
                hist = ticker.history(period="1y")
                if not hist.empty and 'Close' in hist.columns:
                    daily_returns = hist['Close'].pct_change().dropna()
                    implied_volatility = float(daily_returns.std() * np.sqrt(252))
                else:
                    implied_volatility = 0.0
            except Exception as e:
                print(f"Error fetching volatility data: {e}")
                implied_volatility = 0.0
            
            # Historical Operating Margins Average → se calcula con datos ANUALES
            op_margins = []
            if annual_financials is not None and not annual_financials.empty:
                for dt in annual_financials.columns:
                    try:
                        rev = float(annual_financials.loc['Total Revenue', dt])
                        op_inc = float(annual_financials.loc['Operating Income', dt])
                        if rev > 0:
                            op_margins.append(op_inc / rev)
                    except Exception:
                        pass
                        
            historical_avg_op_margin = sum(op_margins) / len(op_margins) if op_margins else 0.10
            
            # Si llegamos aquí, todo salió bien
            return {
                'revenue_base_year': (revenue * exchange_rate_financials) / scale_factor,
                'cash_base_year': (total_cash * exchange_rate_financials) / scale_factor,
                'equity_base_year': (equity * exchange_rate_financials) / scale_factor,
                'debt_base_year': (total_debt * exchange_rate_financials) / scale_factor,
                'interes_expenses': (interest_expense * exchange_rate_financials) / scale_factor,
                'income_base_year': (operating_income * exchange_rate_financials) / scale_factor,
                'minority_interes': (minority_interest * exchange_rate_financials) / scale_factor,
                'non_operating_assets': 0.0, 
                
                'base_r_d_expenses': (rd_expenses_base * exchange_rate_financials) / scale_factor,
                'minus_oneyear_r_d_expense': (rd_m1 * exchange_rate_financials) / scale_factor,
                'minus_twoyear_r_d_expense': (rd_m2 * exchange_rate_financials) / scale_factor,
                'minus_threeyear_r_d_expense': (rd_m3 * exchange_rate_financials) / scale_factor,
                
                'marginal_tax_rate': marginal_tax_rate,
                
                'shares_outstanding': shares_outstanding / scale_factor,  # Shares do not change with currency
                'current_share_price': current_price * exchange_rate_price,
                'beta': beta,
                
                # New Additions
                'industry': industry,
                'country': country,
                'currency': target_currency if target_currency else currency,
                'financialCurrency': target_currency if target_currency else financial_currency,
                'analyst_revenue_growth': analyst_revenue_growth,
                'historical_avg_op_margin': historical_avg_op_margin,
                'implied_volatility': implied_volatility
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            is_auth_or_rate_error = any(x in error_msg for x in [
                "401", "unauthorized", "invalid crumb", "rate limit", "429", "too many requests"
            ])
            
            print(f"⚠️ Error en intento {attempt + 1}/{max_retries} para {ticker_symbol}: {e}")
            
            if attempt < max_retries - 1 and is_auth_or_rate_error:
                wait_time = 1.8 * (attempt + 1)
                print(f"   Reintentando en {wait_time:.1f} segundos...")
                time.sleep(wait_time)
                # Recrear sesión limpia en caso de crumb inválido
                session = requests.Session(impersonate="chrome")
            else:
                print(f"❌ Falló definitivamente la descarga de datos para {ticker_symbol}")
                return None
    
    return None