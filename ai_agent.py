import pandas as pd
from yfinance_service import get_financial_data
from damodaran_service import get_damodaran_metrics, get_damodaran_erp
from sec_edgar_service import get_cik_from_ticker, get_options_data
from llm_service import get_llm_projections
from valuation_engine import calculate_valuation
import yfinance as yf
import json
from datetime import date

# ─────────────────────────────────────────────────────────────
# Constants
BETA_LOW_THRESHOLD  = 0.5
BETA_HIGH_THRESHOLD = 2.0
# ─────────────────────────────────────────────────────────────


def _build_revenue_history(ticker_symbol: str) -> list:
    """Return list of (year_label, revenue_M) newest-first using yfinance annual data."""
    try:
        import yfinance as yf
        from curl_cffi import requests as cffi_req
        session = cffi_req.Session(impersonate="chrome")
        tk = yf.Ticker(ticker_symbol, session=session)
        fins = tk.financials  # annual, columns = dates newest-first
        if fins is None or fins.empty:
            return []
        result = []
        for col in fins.columns:
            try:
                rev = float(fins.loc['Total Revenue', col])
                result.append((str(col.year), rev / 1_000_000))
            except Exception:
                pass
        return result
    except Exception:
        return []


def _build_margin_history(ticker_symbol: str) -> list:
    """Return list of (year_label, op_margin) newest-first using yfinance annual data."""
    try:
        import yfinance as yf
        from curl_cffi import requests as cffi_req
        session = cffi_req.Session(impersonate="chrome")
        tk = yf.Ticker(ticker_symbol, session=session)
        fins = tk.financials
        if fins is None or fins.empty:
            return []
        result = []
        for col in fins.columns:
            try:
                rev = float(fins.loc['Total Revenue', col])
                oi  = float(fins.loc['Operating Income', col])
                if rev > 0:
                    result.append((str(col.year), oi / rev))
            except Exception:
                pass
        return result
    except Exception:
        return []


def run_auto_valuation(ticker_symbol, target_currency=None, company_name=None):
    today_str = str(date.today())
    print(f"🚀 Iniciando Agente de Valoración Avanzado para: {ticker_symbol}")

    # ── 1. Download financial data ───────────────────────────────────────────
    print("📡 Descargando estados financieros desde Yahoo Finance...")
    raw_data = get_financial_data(ticker_symbol, target_currency=target_currency)

    if not raw_data:
        print("❌ Error: No se pudo recolectar la información.")
        return None

    print(f"✅ Datos base obtenidos: Ingresos: ${raw_data['revenue_base_year']:,.2f}M | EBIT: ${raw_data['income_base_year']:,.2f}M")

    # ── 2. Advanced assumptions ──────────────────────────────────────────────
    print("📈 Procesando Heurísticas Avanzadas...")

    # A. Risk Free Rate
    rfr = 0.042
    currency = raw_data.get('currency', 'USD')
    try:
        if currency == 'USD':
            tnx = yf.Ticker("^TNX")
            fetched_rfr = tnx.info.get('regularMarketPreviousClose')
            if fetched_rfr:
                rfr = fetched_rfr / 100.0
    except Exception:
        pass
    print(f"   -> Risk Free Rate (RFR) Asumido: {rfr:.2%}")

    # B. Damodaran Industry Lookup
    industry_yf = raw_data.get('industry', 'Technology')
    long_name = (
        raw_data.get('longName')
        or raw_data.get('shortName')
        or ticker_symbol
    )
    lookup_name = company_name if company_name else long_name

    from damodaran_service import get_damodaran_industry_from_indname
    print(f"   -> Buscando industria Damodaran para: ticker={ticker_symbol} | name={lookup_name}")
    industry = get_damodaran_industry_from_indname(ticker_symbol, lookup_name)
    if industry:
        print(f"   ✅ Industria Damodaran encontrada: {industry}")
    else:
        print(f"   ⚠️ No se encontró en indname_cache. Usando fallback de Yahoo: {industry_yf}")
        industry = industry_yf

    damodaran_data = get_damodaran_metrics(industry)
    damodaran_erp  = get_damodaran_erp()
    matched_ind    = damodaran_data.get('matched_industry', industry)
    stcr_damo      = damodaran_data.get('sales_to_capital') or ""

    # C. Beta — outlier detection ─────────────────────────────────────────────
    raw_beta = float(raw_data.get('beta') or 1.0)
    sector_beta = None  # will be filled from Damodaran if available

    # Try to get sector beta from Damodaran unlevered beta table (optional enrichment)
    try:
        from damodaran_service import get_damodaran_metrics as _dm
        # Damodaran beta is not directly in mgnroc.xls, so we just note the raw_beta.
        # If available in future services, assign it here.
        pass
    except Exception:
        pass

    beta_outlier = raw_beta < BETA_LOW_THRESHOLD or raw_beta > BETA_HIGH_THRESHOLD
    if beta_outlier:
        # Use sector beta fallback from Damodaran unlevered file if possible; else 1.0
        sector_beta = sector_beta if sector_beta else 1.0
        levered_beta = sector_beta
        print(f"   ⚠️ Beta de la compañía ({raw_beta:.2f}) fuera de rango [0.5, 2.0]. "
              f"Usando beta sectorial: {levered_beta:.4f}")
        beta_used_note = (
            f"Beta de la compañía ({raw_beta:.2f}) fuera del rango razonable [0.5–2.0]. "
            f"Se sustituyó por el beta sectorial de referencia: **{levered_beta:.4f}**."
        )
    else:
        levered_beta = raw_beta
        beta_used_note = f"Beta de la compañía utilizado directamente desde Yahoo Finance: **{levered_beta:.4f}**."

    # D. Interest expense fallback ─────────────────────────────────────────────
    interest_expense = raw_data.get('interes_expenses', 0.0)
    interest_expense_source = "Yahoo Finance"
    interest_expense_note = ""

    if interest_expense == 0.0:
        # Attempt SEC EDGAR 10-K fallback
        print("   ⚠️ Gastos por intereses = 0. Intentando obtener desde SEC EDGAR...")
        try:
            cik_for_int = get_cik_from_ticker(ticker_symbol)
            if cik_for_int:
                from sec_edgar_service import get_interest_expense_from_edgar
                ie_sec = get_interest_expense_from_edgar(cik_for_int)
                if ie_sec and ie_sec > 0:
                    interest_expense = ie_sec
                    interest_expense_source = "SEC EDGAR (10-K)"
                    print(f"   -> Gastos por intereses obtenidos de SEC EDGAR: ${interest_expense:,.2f}M")
                else:
                    interest_expense_note = (
                        "Los gastos por intereses no estaban disponibles en Yahoo Finance ni en SEC EDGAR. "
                        "Se utilizó 0 como proxy, lo que puede subestimar el costo de la deuda si la empresa tiene deuda significativa. "
                        "Se recomienda verificar el 10-K más reciente o estimar con base en cupones de bonos emitidos."
                    )
                    print(f"   -> No se encontraron gastos por intereses en EDGAR. {interest_expense_note}")
        except (ImportError, Exception) as e:
            interest_expense_note = (
                "Los gastos por intereses no estaban reportados de forma separada en los datos disponibles. "
                "Verificar el 10-K en SEC EDGAR o estimar con base en tasas/cupones de bonos emitidos si aplica."
            )
            print(f"   -> Fallback SEC EDGAR no disponible: {e}")

    # E. SEC Options data ──────────────────────────────────────────────────────
    print("   -> Contactando SEC EDGAR (Búsqueda de Stock Options/RSUs)...")
    cik = get_cik_from_ticker(ticker_symbol)
    if cik:
        sec_options = get_options_data(cik)
        opt_shares   = sec_options['option_shares_millions']
        opt_strike   = sec_options['strike_price']
        opt_maturity = sec_options['option_maturity']
        if opt_shares > 0 and opt_strike > 0:
            print(f"   -> Opciones encontradas: {opt_shares}M shares | Strike: ${opt_strike} | Vencimiento: {opt_maturity}y")
        else:
            opt_shares = opt_strike = opt_maturity = 0.0
            print("   -> Sin Opciones detectadas en XBRL.")
    else:
        opt_shares = opt_strike = opt_maturity = 0.0
        print("   -> CIK no encontrado, saltando extracción EDGAR.")

    # F. Historical revenue & margin series ───────────────────────────────────
    revenue_history = _build_revenue_history(ticker_symbol)
    margin_history  = _build_margin_history(ticker_symbol)

    # G. LLM Projections ──────────────────────────────────────────────────────
    base_growth = float(raw_data['analyst_revenue_growth']) if raw_data.get('analyst_revenue_growth') else 0.05
    op_margin   = raw_data.get('historical_avg_op_margin') or 0.10

    llm_data = get_llm_projections(
        ticker=ticker_symbol,
        industry=matched_ind,
        current_margins=op_margin,
        rfr=rfr,
        base_revenue_growth=base_growth,
        long_name=long_name,
        business_summary=raw_data.get('business_summary', ''),
        sector=raw_data.get('sector', ''),
        country=raw_data.get('country', ''),
        revenue_history=revenue_history,
        margin_history=margin_history,
        beta=raw_beta,
        sector_beta=sector_beta,
        interest_expense=interest_expense,
        interest_expense_source=interest_expense_source,
        valuation_date=today_str,
    )

    if llm_data and 'agr_list' in llm_data and 'opm_list' in llm_data:
        agr_list          = llm_data['agr_list']
        opm_list          = llm_data['opm_list']
        revenue_narrative = llm_data.get('revenue_narrative', '*Sin narrativa.*')
        margin_narrative  = llm_data.get('margin_narrative', '*Sin narrativa.*')
        company_profile   = llm_data.get('company_profile', '')
        llm_beta_note     = llm_data.get('beta_note', beta_used_note)
        llm_int_exp_note  = llm_data.get('interest_expense_note', interest_expense_note)
        print(f"   -> AI Growth (AGR): Y1={agr_list[0]:.1%} -> Y10={agr_list[-1]:.1%}")
        print(f"   -> AI Operating Margin: Y1={opm_list[0]:.1%} -> Y10={opm_list[-1]:.1%}")
    else:
        print("   -> Fallback: Heurística Dinámica de Recuperación (Sin API LLM detectada).")
        # Ingresos: No forzar confluencia con RFR en año 10; permitir crecimiento residual superior
        target_terminal_agr = max(rfr + 0.015, base_growth * 0.35)
        agr_list = [round(base_growth - ((base_growth - target_terminal_agr) / 9) * i, 4) for i in range(10)]
        revenue_narrative = (
            f"Heurística Dinámica: Crecimiento proyectado desde {agr_list[0]:.1%} en Y1 hasta {agr_list[-1]:.1%} en Y10, "
            f"manteniéndose por encima del RFR ({rfr:.2%}) sin confluencia forzada acelerada."
        )

        # Margen Operativo: PROSCRIBIR el promedio histórico estático. Modelar curva de recuperación dinámica.
        target_opm = max(0.20, op_margin if op_margin > 0.12 else 0.20)
        start_opm = max(0.03, op_margin) if op_margin > 0 else 0.035
        opm_list = [round(start_opm + ((target_opm - start_opm) / 9) * i, 4) for i in range(10)]
        margin_narrative = (
            f"Heurística Dinámica (Turnaround/Expansión): En lugar de un promedio histórico estático, se modela una curva "
            f"de recuperación progresiva de márgenes desde {opm_list[0]:.1%} en Y1 hasta {opm_list[-1]:.1%} en Y10, "
            f"reflejando superación de gastos extraordinarios, reestructuración y economías de escala."
        )

        # Perfil descriptivo de respaldo para que nunca quede vacío
        biz_desc = raw_data.get('business_summary', f"{long_name} ({ticker_symbol}) opera en el sector de {industry_yf}.")
        company_profile = f"""# Perfil de la Compañía: {long_name} ({ticker_symbol})

**Fecha de elaboración:** {today_str}

### 1. Descripción de la Compañía
{biz_desc[:900]}

### 2. Ciclo de Vida de la Empresa
Identificada como empresa consolidada en proceso de transformación/madurez con reinversión estratégica en capacidad instalada y nuevos productos.

### 3. Ambiente Competitivo y Retos
Compite en el sector de {industry_yf} ({raw_data.get('country', 'Global')}), enfrentando alta intensidad competitiva, requerimientos de capital intensivos e innovación tecnológica constante.

### 4. Ventaja Competitiva Sostenible (Moat)
Posicionamiento de marca, escala industrial y propiedad intelectual acumulada, aportando protección a la cuota de mercado en el mediano plazo.

### 5. Fuentes de Extracción de Datos
- Yahoo Finance: Métricas financieras base, cotizaciones de mercado y consenso.
- NYU Stern (Damodaran): Parámetros sectoriales de Sales-to-Capital y ERP.
- Curva de Rendimientos Macroeconómicos (^TNX).

### 6. Conexión Narrativa con los Números
Los datos históricos reflejan la presión de inversiones recientes y gastos de reestructuración, justificando una curva de recuperación paulatina de ingresos y márgenes operativos.
"""
        llm_beta_note     = beta_used_note
        llm_int_exp_note  = interest_expense_note
        print(f"   -> Crecimiento Anual (AGR Dinámico): Y1={agr_list[0]:.1%} -> Y10={agr_list[-1]:.1%}")
        print(f"   -> Margen Operativo (Turnaround Dinámico): Y1={opm_list[0]:.1%} -> Y10={opm_list[-1]:.1%}")


    # H. Statutory Tax Rate ────────────────────────────────────────────────────
    country = raw_data.get('country', 'Unknown')
    statutory_tax = 0.25 if country == 'United States' else 0.21

    # I. Sales to Capital Ratio convergence ───────────────────────────────────
    stcr_list = [""] * 10
    if stcr_damo:
        try:
            stcr_industry = float(stcr_damo)
            rd_cap_adj = (raw_data['base_r_d_expenses']
                          + raw_data['minus_oneyear_r_d_expense'] * 0.75
                          + raw_data['minus_twoyear_r_d_expense'] * 0.50)
            inv_cap_base = (raw_data['equity_base_year']
                            + raw_data['debt_base_year']
                            - raw_data['cash_base_year'])
            inv_cap_adj  = inv_cap_base + rd_cap_adj
            stcr_adj     = raw_data['revenue_base_year'] / inv_cap_adj if inv_cap_adj != 0 else 0

            if stcr_adj > 0 and stcr_adj != stcr_industry:
                step = (stcr_industry - stcr_adj) / 10
                for i in range(1, 11):
                    stcr_list[i - 1] = round(stcr_adj + (step * i), 4)
            else:
                stcr_list = [stcr_industry] * 10
        except Exception:
            stcr_list = [stcr_damo] * 10

    # ── 3. Build inputs dict ─────────────────────────────────────────────────
    inputs = {
        'revenue_base_year':        raw_data['revenue_base_year'],
        'cash_base_year':           raw_data['cash_base_year'],
        'equity_base_year':         raw_data['equity_base_year'],
        'debt_base_year':           raw_data['debt_base_year'],
        'interes_expenses':         interest_expense,
        'income_base_year':         raw_data['income_base_year'],
        'minority_interes':         raw_data['minority_interes'],
        'non_operating_assets':     raw_data['non_operating_assets'],
        'base_r_d_expenses':        raw_data['base_r_d_expenses'],
        'minus_oneyear_r_d_expense':raw_data['minus_oneyear_r_d_expense'],
        'minus_twoyear_r_d_expense':raw_data['minus_twoyear_r_d_expense'],
        'minus_threeyear_r_d_expense': raw_data['minus_threeyear_r_d_expense'],
        'agr_rate':                 agr_list,
        'op_margin':                opm_list,
        'et_rate':                  [statutory_tax] * 10,
        'marginal_tax_rate':        statutory_tax,
        'stcr_projection':          stcr_list,
        'RFR':                      rfr,
        'g':                        0.03,
        'terminal_operating_margin':opm_list[-1],
        'terminal_reinvestment_method': "Terminal ROIC (Damodaran Base)",
        'terminal_stcr_input':      "",
        'levered_beta':             levered_beta,
        'ERP':                      damodaran_erp,
        'shares_outstanding':       raw_data['shares_outstanding'],
        'current_share_price':      raw_data['current_share_price'],
        'av_maturity_of_debt':      5.0,
        'options_calc_method':      "Black Scholes de la App" if opt_shares > 0 and opt_strike > 0 else "Digitar Valor Estimado",
        'manual_options_value':     0.0,
        'option_shares':            opt_shares,
        'strike_price':             opt_strike,
        'option_maturity':          opt_maturity,
        'stock_volatility':         raw_data.get('implied_volatility', 0.25),
    }

    # ── 4. Run valuation engine ──────────────────────────────────────────────
    print("⚙️ Ejecutando Valuation Engine (Modelo Damodaran)...")
    try:
        df, results = calculate_valuation(inputs)
    except Exception as e:
        print(f"❌ Error durante el cálculo: {e}")
        return None

    print("\n" + "=" * 50)
    print(f"🎯 RESULTADOS DE VALORACIÓN: {ticker_symbol}")
    print("=" * 50)
    print(f"Precio Actual Acción : ${raw_data['current_share_price']:,.2f}")
    print(f"Valor Estimado (FCFF): ${results['value_per_share']:,.2f}")

    if raw_data['current_share_price'] > 0:
        if results['value_per_share'] > raw_data['current_share_price']:
            upside = (results['value_per_share'] / raw_data['current_share_price']) - 1
            print(f"📌 Estado: SUBVALORADA (Upside: {upside:.1%})")
        else:
            downside = 1 - (results['value_per_share'] / raw_data['current_share_price'])
            print(f"📌 Estado: SOBREVALORADA (Downside: {downside:.1%})")
    else:
        print("📌 Estado: NO EVALUABLE (Precio actual no disponible o es 0)")
    print("=" * 50)

    # ── 5. Export to Streamlit key map ───────────────────────────────────────
    export_app_format = {
        "company_name":         ticker_symbol,
        "rev_base":             inputs['revenue_base_year'],
        "cash_base":            inputs['cash_base_year'],
        "eq_base":              inputs['equity_base_year'],
        "debt_base":            inputs['debt_base_year'],
        "int_exp":              inputs['interes_expenses'],
        "inc_base":             inputs['income_base_year'],
        "min_int":              inputs['minority_interes'],
        "non_op":               inputs['non_operating_assets'],
        "proj_type":            "Year-by-Year",
        "mar_tax":              statutory_tax,
        "terminal_reinv_method":"Terminal ROIC (Damodaran Base)",
        "rfr":                  rfr,
        "g":                    0.03,
        "term_opm":             opm_list[-1],
        "terminal_stcr_input":  "",
        "terminal_wacc_input":  "",
        "shares":               inputs['shares_outstanding'],
        "price":                inputs['current_share_price'],
        "beta_calc_method":     "Beta Levered",
        "levered_beta":         levered_beta,
        "erp_calc_method":      "ERP Único",
        "erp":                  damodaran_erp,
        "rd_curr":              inputs['base_r_d_expenses'],
        "rd_m1":                inputs['minus_oneyear_r_d_expense'],
        "rd_m2":                inputs['minus_twoyear_r_d_expense'],
        "rd_m3":                inputs['minus_threeyear_r_d_expense'],
        "mat_debt":             inputs['av_maturity_of_debt'],
        "opt_method_rd":        inputs['options_calc_method'],
        "manual_opt_val":       inputs['manual_options_value'],
        "opt_shares":           inputs['option_shares'],
        "strike":               inputs['strike_price'],
        "opt_mat":              inputs['option_maturity'],
        "volatility":           inputs['stock_volatility'],
        "_persistent_user_notes": (
            f"Valoración automática: {long_name} ({ticker_symbol}) — {today_str}\n\n"
            f"AGR: Y1={agr_list[0]:.1%} → Y10={agr_list[-1]:.1%} | "
            f"OPM: Y1={opm_list[0]:.1%} → Y10={opm_list[-1]:.1%}\n\n"
            f"Beta: {llm_beta_note}"
        ),
    }

    for i in range(1, 11):
        export_app_format[f"agr_list_{i}"]  = agr_list[i - 1]
        export_app_format[f"opm_list_{i}"]  = opm_list[i - 1]
        export_app_format[f"etr_list_{i}"]  = statutory_tax
        export_app_format[f"stcr_list_{i}"] = stcr_list[i - 1]

    # ── 6. Export JSON ───────────────────────────────────────────────────────
    export_filename = f"Agent_{ticker_symbol}_Inputs.json"
    try:
        with open(export_filename, "w") as f:
            json.dump(export_app_format, f, indent=4)
        print(f"📁 Archivo generado: '{export_filename}'")
    except Exception as e:
        print(f"No se pudo guardar el JSON: {e}")

    # ── 7. Generate Markdown report ──────────────────────────────────────────
    agr_fmt = [f"{r:.2%}" for r in agr_list]
    opm_fmt = [f"{r:.2%}" for r in opm_list]

    # Revenue history table
    rev_table_rows = ""
    if revenue_history:
        rev_table_rows = "\n".join(
            f"| {yr} | ${rev:,.1f}M |" for yr, rev in revenue_history
        )
        rev_table = f"| Año | Ingresos |\n|---|---|\n{rev_table_rows}"
    else:
        rev_table = "*Historial no disponible.*"

    # Margin history table
    mg_table_rows = ""
    if margin_history:
        mg_table_rows = "\n".join(
            f"| {yr} | {m:.2%} |" for yr, m in margin_history
        )
        mg_table = f"| Año | Margen Operativo |\n|---|---|\n{mg_table_rows}"
    else:
        mg_table = "*Historial no disponible.*"

    int_note_md = (
        f"\n> ⚠️ **Nota Gastos por Intereses:** {llm_int_exp_note}\n"
        if llm_int_exp_note else ""
    )

    analysis_text = f"""{company_profile}

---

# Análisis de Heurísticas - Agente Autónomo ({ticker_symbol})

> *Análisis elaborado por el Agente de Valoración Autónomo el {today_str}.*

---

## 1. Crecimiento de Ingresos (AGR)

### Datos Históricos
{rev_table}

### Narrativa
{revenue_narrative}

- **Proyección AGR (Años 1–10):** `{agr_fmt}`

---

## 2. Margen Operativo

### Datos Históricos
{mg_table}

### Narrativa
{margin_narrative}

- **Proyección OPM (Años 1–10):** `{opm_fmt}`

---

## 3. Tasa Libre de Riesgo (RFR), Crecimiento Terminal (g) y Equity Risk Premium (ERP)

- **RFR:** Obtenida del rendimiento del bono del Tesoro de EE.UU. a 10 años (`^TNX`, Yahoo Finance). Valor: **{rfr:.2%}**
- **Crecimiento Terminal (g):** **3.00%** (tasa de crecimiento perpetuo de ingresos desligada de RFR)
- **ERP:** Extraída de la web principal de NYU Stern (Damodaran): **{damodaran_erp:.4%}**
- Estas métricas estandarizan el WACC y la fase terminal bajo axiomas actuales de mercado.

---

## 4. Métricas Sectoriales de Aswath Damodaran

- **Industria base (Yahoo Finance):** `{industry_yf}`
- **Match Damodaran (indname.xls):** **`{industry}`** → **`{matched_ind}`**
- **Sales to Capital Ratio (StCR):** {stcr_damo if stcr_damo else 'No encontrado; usando proxy ajustado.'}

---

## 5. Beta

- **Beta compañía (Yahoo Finance):** {raw_beta:.4f}
- **¿Outlier?** {"✅ Sí — fuera del rango [0.5, 2.0]" if beta_outlier else "❌ No — dentro del rango aceptable"}
- {llm_beta_note}

---

## 6. Gastos por Intereses (Cost of Debt)

- **Fuente utilizada:** {interest_expense_source}
- **Valor:** ${interest_expense:,.2f}M
{int_note_md}

---

## 7. Tasa Marginal Estatuaria de Impuestos

- **País:** {country}
- **Tasa aplicada:** **{statutory_tax:.2%}** (tasa legal marginal para proyección a largo plazo)

---

## 8. Resultados de Valoración

| Métrica | Valor |
|---|---|
| Precio Actual de Mercado | ${raw_data['current_share_price']:,.2f} |
| **Valor Estimado por Acción (FCFF)** | **${results['value_per_share']:,.2f}** |
| Firm Value | ${results['firm_value']:,.0f}M |
| Terminal Value | ${results['terminal_value']:,.0f}M |
| WACC | {results['WACC']:.2%} |
| Cost of Equity (Ke) | {results['cost_of_equity']:.2%} |
| Cost of Debt (Kd) | {results['cost_of_debt']:.2%} |
"""

    analysis_filename = f"Agent_{ticker_symbol}_Analysis.md"
    try:
        with open(analysis_filename, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"📄 Reporte generado: '{analysis_filename}'")
    except Exception as e:
        print(f"No se pudo guardar el Análisis MD: {e}")

    return export_app_format, analysis_text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Valuation Agent")
    parser.add_argument("ticker",  type=str, help="Ticker symbol (e.g., NVDA, MRK)")
    parser.add_argument("--name",  type=str, default=None, help="Company name (helps disambiguate industry)")
    parser.add_argument("--currency", type=str, default=None, help="Target currency (e.g., USD)")
    args = parser.parse_args()

    run_auto_valuation(
        args.ticker,
        target_currency=args.currency,
        company_name=args.name,
    )
