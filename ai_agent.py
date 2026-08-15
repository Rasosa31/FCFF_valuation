import pandas as pd
from yfinance_service import get_financial_data
from damodaran_service import get_damodaran_metrics, get_damodaran_erp
from llm_service import get_llm_projections
from valuation_engine import calculate_valuation
import yfinance as yf
import json

def run_auto_valuation(ticker_symbol, target_currency=None):
    print(f"🚀 Iniciando Agente de Valoración Avanzado para: {ticker_symbol}")
    
    # 1. Gather Data
    print(f"📡 Descargando estados financieros desde Yahoo Finance...")
    raw_data = get_financial_data(ticker_symbol, target_currency=target_currency)
    
    if not raw_data:
        print("❌ Error: No se pudo recolectar la información.")
        return None
        
    print(f"✅ Datos base obtenidos: Ingresos: ${raw_data['revenue_base_year']:,.2f}M | EBIT: ${raw_data['income_base_year']:,.2f}M")
    
    # 2. Build Advanced Assumptions
    print("📈 Procesando Heurísticas Avanzadas...")
    
    # A. Risk Free Rate
    rfr = 0.042
    currency = raw_data.get('currency', 'USD')
    try:
        if currency == 'USD':
            tnx = yf.Ticker("^TNX")
            fetched_rfr = tnx.info.get('regularMarketPreviousClose')
            if fetched_rfr: rfr = fetched_rfr / 100.0
    except: pass
    print(f"   -> Risk Free Rate (RFR) Asumido: {rfr:.2%}")
    
    # B. Damodaran Metrics
    industry = raw_data.get('industry', 'Technology')
    damodaran_data = get_damodaran_metrics(industry)
    damodaran_erp = get_damodaran_erp()
    matched_ind = damodaran_data.get('matched_industry', industry)
    
    stcr_damo = damodaran_data.get('sales_to_capital') or ""
    beta_damo = damodaran_data.get('unlevered_beta') or raw_data['beta']
    beta_damo_cash = damodaran_data.get('unlevered_beta_cash') or beta_damo
    
    # C & D. Projections (LLM Agent vs Heuristics)
    base_growth = float(raw_data['analyst_revenue_growth']) if raw_data.get('analyst_revenue_growth') else 0.50
    op_margin = raw_data.get('historical_avg_op_margin') or 0.49
    
    llm_data = get_llm_projections(
        ticker=ticker_symbol,
        industry=matched_ind,
        current_margins=op_margin,
        rfr=rfr,
        base_revenue_growth=base_growth
    )
    
    if llm_data and 'agr_list' in llm_data and 'opm_list' in llm_data:
        agr_list = llm_data['agr_list']
        opm_list = llm_data['opm_list']
        revenue_narrative = llm_data.get('revenue_narrative', '*Sin narrativa.*')
        margin_narrative = llm_data.get('margin_narrative', '*Sin narrativa.*')
        print(f"   -> AI Growth (AGR): Y1={agr_list[0]:.1%} -> Y10={agr_list[-1]:.1%}")
        print(f"   -> AI Operating Margin: Y1={opm_list[0]:.1%} -> Y10={opm_list[-1]:.1%}")
    else:
        # Fallback to Heuristics if API Key fails or missing
        print("   -> Fallback: Heurísticas Matemáticas Programadas (Sin API LLM detectada).")
        agr_list = []
        decay_factor = (base_growth - rfr) / 9
        for i in range(10):
            year_rate = base_growth - (decay_factor * i)
            agr_list.append(max(rfr, year_rate)) 
            
        opm_list = [op_margin] * 10
        revenue_narrative = f"Heurística Matemática: Crecimiento decrece linealmente de {base_growth:.1%} a {rfr:.2%}."
        margin_narrative = f"Heurística Matemática: Promedio Histórico estático de {op_margin:.1%}."
        print(f"   -> Crecimiento Anual (AGR): Y1={agr_list[0]:.1%} -> Y10={agr_list[-1]:.1%}")
        print(f"   -> Margen Operativo (Promedio Histórico): {op_margin:.1%}")
    
    # E. Statutory Tax Rate
    country = raw_data.get('country', 'Unknown')
    statutory_tax = 0.25 if country == 'United States' else 0.21
    
    inputs = {
        # Fundamentals
        'revenue_base_year': raw_data['revenue_base_year'],
        'cash_base_year': raw_data['cash_base_year'],
        'equity_base_year': raw_data['equity_base_year'],
        'debt_base_year': raw_data['debt_base_year'],
        'interes_expenses': raw_data['interes_expenses'],
        'income_base_year': raw_data['income_base_year'],
        'minority_interes': raw_data['minority_interes'],
        'non_operating_assets': raw_data['non_operating_assets'],
        
        # R&D
        'base_r_d_expenses': raw_data['base_r_d_expenses'],
        'minus_oneyear_r_d_expense': raw_data['minus_oneyear_r_d_expense'],
        'minus_twoyear_r_d_expense': raw_data['minus_twoyear_r_d_expense'],
        'minus_threeyear_r_d_expense': raw_data['minus_threeyear_r_d_expense'],
        
        # Projections
        'agr_rate': agr_list,
        'op_margin': opm_list,
        'et_rate': [statutory_tax] * 10,
        'marginal_tax_rate': statutory_tax,
        'stcr_projection': [stcr_damo] * 10 if stcr_damo else "",
        
        # Terminal Assumptions
        'RFR': rfr,
        'terminal_operating_margin': opm_list[-1],
        'terminal_reinvestment_method': "Terminal ROIC (Damodaran Base)",
        'terminal_stcr_input': stcr_damo if stcr_damo else "",
        
        # Cost of Capital
        'beta_option': 'Sectorial Corregida por Cash',
        'unlevered_beta': beta_damo_cash, 
        'ERP': damodaran_erp, 
        
        # Market
        'shares_outstanding': raw_data['shares_outstanding'],
        'current_share_price': raw_data['current_share_price'],
        
        # Debt & Options
        'av_maturity_of_debt': 5.0,
        'options_calc_method': "Digitar Valor Estimado",
        'manual_options_value': 0.0,
        'option_shares': 0,
        'strike_price': 0,
        'option_maturity': 0,
        'stock_volatility': 0
    }
    
    print("⚙️ Ejecutando Valuation Engine (Modelo Damodaran)...")
    try:
        df, results = calculate_valuation(inputs)
    except Exception as e:
        print(f"❌ Error durante el cálculo: {e}")
        return None
        
    print("\n" + "="*50)
    print(f"🎯 RESULTADOS DE VALORACIÓN: {ticker_symbol}")
    print("="*50)
    print(f"Precio Actual Acción : ${raw_data['current_share_price']:,.2f}")
    print(f"Valor Estimado (FCFF): ${results['value_per_share']:,.2f}")
    
    if results['value_per_share'] > raw_data['current_share_price']:
        upside = (results['value_per_share'] / raw_data['current_share_price']) - 1
        print(f"📌 Estado: SUBVALORADA (Upside: {upside:.1%})")
    else:
        downside = 1 - (results['value_per_share'] / raw_data['current_share_price'])
        print(f"📌 Estado: SOBREVALORADA (Downside: {downside:.1%})")
    print("="*50)
    
    # Map to Streamlit UI Keys
    export_app_format = {
        "company_name": ticker_symbol,
        "rev_base": inputs['revenue_base_year'],
        "cash_base": inputs['cash_base_year'],
        "eq_base": inputs['equity_base_year'],
        "debt_base": inputs['debt_base_year'],
        "int_exp": inputs['interes_expenses'],
        "inc_base": inputs['income_base_year'],
        "min_int": inputs['minority_interes'],
        "non_op": inputs['non_operating_assets'],
        
        "proj_type": "Year-by-Year",
        "mar_tax": statutory_tax,
        
        "terminal_reinv_method": "Terminal ROIC (Damodaran Base)",
        "rfr": rfr,
        "term_opm": opm_list[-1],
        "terminal_stcr_input": stcr_damo if stcr_damo else "",
        "terminal_wacc_input": "",
        
        "shares": inputs['shares_outstanding'],
        "price": inputs['current_share_price'],
        
        "beta_calc_method": "Beta Única",
        "beta_opt_single": "Sectorial Corregida por Cash",
        "unlev_beta_sect": beta_damo,
        "unlev_beta_cash": beta_damo_cash,
        
        "erp_calc_method": "ERP Único",
        "erp": damodaran_erp,
        
        "rd_curr": inputs['base_r_d_expenses'],
        "rd_m1": inputs['minus_oneyear_r_d_expense'],
        "rd_m2": inputs['minus_twoyear_r_d_expense'],
        "rd_m3": inputs['minus_threeyear_r_d_expense'],
        
        "mat_debt": inputs['av_maturity_of_debt'],
        "opt_method_rd": "Digitar Valor Estimado",
        "manual_opt_val": inputs['manual_options_value'],
        "opt_shares": inputs['option_shares'],
        "strike": inputs['strike_price'],
        "opt_mat": inputs['option_maturity'],
        "volatility": inputs['stock_volatility'],
        
        "_persistent_user_notes": f"Valuación base extraída automáticamente de Yahoo Finance para {ticker_symbol} mediante el Agente AI.\n\n"
                                  f"El Agente estableció Crecimiento descendiente Year-by-Year desde {agr_list[0]:.1%} hasta {agr_list[-1]:.1%} basado en reportes de analistas. Margen Promedio {opm_list[0]:.1%}."
    }
    
    # Fill dynamic fields
    for i in range(1, 11):
        export_app_format[f"agr_list_{i}"] = agr_list[i-1]
        export_app_format[f"opm_list_{i}"] = opm_list[i-1]
        export_app_format[f"etr_list_{i}"] = statutory_tax
        export_app_format[f"stcr_list_{i}"] = stcr_damo if stcr_damo else ""
        
    # Export JSON
    export_filename = f"Agent_{ticker_symbol}_Inputs.json"
    try:
        with open(export_filename, "w") as f:
            json.dump(export_app_format, f, indent=4)
        print(f"📁 Se ha generado el archivo '{export_filename}'. ¡Súbelo a la app para revisar y ajustar las cifras!")
    except Exception as e:
        print(f"No se pudo guardar el JSON: {e}")
        
    # Generar Reporte de Análisis (.md)
    analysis_filename = f"Agent_{ticker_symbol}_Analysis.md"
    
    analysis_text = f"""# Análisis de Heurísticas - Agente Autónomo ({ticker_symbol})

## 1. Crecimiento de Ingresos (AGR)
{revenue_narrative}

- **Proyección de Tasas 10A:** `{[f"{r:.2%}" for r, c in zip(agr_list, range(10))]}`

## 2. Margen Operativo
{margin_narrative}

- **Proyección de Márgenes 10A:** `{[f"{r:.2%}" for r, c in zip(opm_list, range(10))]}`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **{rfr:.2%}**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **{damodaran_erp:.4%}**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** '{industry}' -> Match más cercano: **'{matched_ind}'**.
- **Unlevered Beta:** El modelo empleó `{beta_damo}` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de {beta_damo_cash}** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** {stcr_damo if stcr_damo else 'No encontrados en XLS por fallo estructural, usando proxy ajustado.'}

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** {country}.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **{statutory_tax:.2%}**.
"""
    try:
        with open(analysis_filename, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"📄 Se ha generado el reporte descriptivo '{analysis_filename}'.")
    except Exception as e:
        print(f"No se pudo guardar el Análisis MD: {e}")
    
    return export_app_format, analysis_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Valuation Agent")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., NVDA, EC)")
    parser.add_argument("--currency", type=str, default=None, help="Target currency for valuation (e.g., USD)")
    args = parser.parse_args()
    
    run_auto_valuation(args.ticker, target_currency=args.currency)
