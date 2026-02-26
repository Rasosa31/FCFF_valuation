import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from valuation_engine import calculate_valuation, run_montecarlo_sim

def to_excel(df, results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Projections')
        pd.DataFrame(list(results.items()), columns=['Metric', 'Value']).to_excel(writer, index=False, sheet_name='Valuation Metrics')
    return output.getvalue()

st.set_page_config(page_title="FCFF Valuation App", layout="wide")

st.title("Valuation App - Free Cash Flow to Firm (FCFF)")
st.markdown("Basado en metodologías de valoración de Aswath Damodaran")

st.sidebar.header("Input Variables")

if st.sidebar.button("Clear Data"):
    st.session_state.clear()
    st.rerun()

st.sidebar.subheader("0. General Info")
company_name = st.sidebar.text_input("Company Name", "My Company")
valuation_date = st.sidebar.date_input("Valuation Date")

def float_input(label, default_val, format="%.2f"):
    return st.sidebar.number_input(label, value=float(default_val), format=format)

st.sidebar.subheader("1. Base Data")
revenue_base_year = float_input("Revenue Base Year", 175294)
cash_base_year = float_input("Cash Base Year", 38066)
equity_base_year = float_input("Equity Base Year (Book Value)", 82324)
debt_base_year = float_input("Debt Base Year", 31454)
interes_expenses = float_input("Interest Expenses", 1258)
income_base_year = float_input("Income Base Year (Operating Income)", 17186)
minority_interes = float_input("Minority Interest", 599)
non_operating_assets = float_input("Non Operating Assets", 0)

st.sidebar.subheader("2. Projections & Rates (1-10 Years)")
proj_type = st.sidebar.radio("Input Method for Rates", ["Single Value", "Year-by-Year"])

if proj_type == "Single Value":
    agr_rate = float_input("Annual Revenue Growth Rate", 0.05, format="%.4f")
    op_margin = float_input("Operating Margin", 0.10, format="%.4f")
    et_rate = float_input("Effective Tax Rate", 0.14, format="%.4f")
    stcr_projection = float_input("Sales to Capital Ratio (StCR) Projection", 0.8, format="%.2f")
else:
    agr_rate = []
    op_margin = []
    et_rate = []
    stcr_projection = []
    st.sidebar.markdown("**Annual Revenue Growth Rate**")
    for i in range(1, 11): agr_rate.append(st.sidebar.number_input(f"AGR Year {i}", value=0.05, format="%.4f", key=f"agr_{i}"))
    st.sidebar.markdown("**Operating Margin**")
    for i in range(1, 11): op_margin.append(st.sidebar.number_input(f"Op Margin Year {i}", value=0.10, format="%.4f", key=f"op_{i}"))
    st.sidebar.markdown("**Effective Tax Rate**")
    for i in range(1, 11): et_rate.append(st.sidebar.number_input(f"Tax Rate Year {i}", value=0.14, format="%.4f", key=f"tax_{i}"))
    st.sidebar.markdown("**Sales to Capital Ratio (StCR)**")
    for i in range(1, 11): stcr_projection.append(st.sidebar.number_input(f"StCR Year {i}", value=0.80, format="%.2f", key=f"stcr_{i}"))

marginal_tax_rate = float_input("Marginal Tax Rate", 0.25, format="%.4f")

st.sidebar.subheader("3. Terminal Year")
RFR = float_input("Risk Free Rate (RFR) / Terminal Growth", 0.0445, format="%.4f")
terminal_operating_margin = float_input("Terminal Operating Margin", 0.10, format="%.4f")
terminal_wacc_input = st.sidebar.text_input("Terminal WACC (leave empty to use current)", "")

st.sidebar.subheader("4. Market & Equity")
shares_outstanding = float_input("Shares Outstanding", 2941.6)
current_share_price = float_input("Current Share Price", 12.7)
levered_beta = float_input("Levered Beta", 1.24)
ERP = float_input("Equity Risk Premium (ERP)", 0.0429, format="%.4f")

st.sidebar.subheader("5. R&D Expenses")
base_r_d_expenses = float_input("Current Year R&D", 5392)
minus_oneyear_r_d_expense = float_input("R&D (T-1)", 5493)
minus_twoyear_r_d_expense = float_input("R&D (T-2)", 5122)
minus_threeyear_r_d_expense = float_input("R&D (T-3)", 4336)

st.sidebar.subheader("6. Debt & Options")
av_maturity_of_debt = float_input("Average Maturity of Debt (Years)", 5, format="%.1f")
option_shares = float_input("Employee Options (Shares)", 0)
strike_price = float_input("Average Strike Price", 0)
option_maturity = float_input("Average Option Maturity (Years)", 0)
stock_volatility = float_input("Stock Volatility (e.g. 0.3 for 30%)", 0.0, format="%.4f")

# Build Input Dictionary
inputs = {
    'revenue_base_year': revenue_base_year,
    'cash_base_year': cash_base_year,
    'equity_base_year': equity_base_year,
    'debt_base_year': debt_base_year,
    'interes_expenses': interes_expenses,
    'income_base_year': income_base_year,
    'minority_interes': minority_interes,
    'non_operating_assets': non_operating_assets,
    
    'agr_rate': agr_rate,
    'op_margin': op_margin,
    'et_rate': et_rate,
    'stcr_projection': stcr_projection,
    'marginal_tax_rate': marginal_tax_rate,
    
    'RFR': RFR,
    'terminal_operating_margin': terminal_operating_margin,
    
    'shares_outstanding': shares_outstanding,
    'current_share_price': current_share_price,
    'levered_beta': levered_beta,
    'ERP': ERP,
    
    'base_r_d_expenses': base_r_d_expenses,
    'minus_oneyear_r_d_expense': minus_oneyear_r_d_expense,
    'minus_twoyear_r_d_expense': minus_twoyear_r_d_expense,
    'minus_threeyear_r_d_expense': minus_threeyear_r_d_expense,
    
    'av_maturity_of_debt': av_maturity_of_debt,
    'option_shares': option_shares,
    'strike_price': strike_price,
    'option_maturity': option_maturity,
    'stock_volatility': stock_volatility
}

if terminal_wacc_input.strip() != "":
    try:
        inputs['terminal_wacc'] = float(terminal_wacc_input)
    except ValueError:
        st.sidebar.error("Terminal WACC must be a number.")

# Add a Calculate Button
if st.sidebar.button("Run Valuation"):
    with st.spinner("Calculating..."):
        df, results = calculate_valuation(inputs)
        st.session_state['df'] = df
        st.session_state['results'] = results

if 'df' in st.session_state and 'results' in st.session_state:
    df = st.session_state['df']
    results = st.session_state['results']
    
    tab1, tab2, tab3, tab8, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard", "Cashflow Projections", "Montecarlo Simulation", "Detalle y Valor (Bridge)",
        "WACC Detail", "Sales to Capital Detail", "R&D Adjustment Detail", "Value Options Detail"
    ])
    
    with tab1:
        st.header(f"Valuation Summary: {company_name}")
        st.write(f"Date: {valuation_date.strftime('%Y-%m-%d')}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Value per Share", f"${results['value_per_share']:,.2f}")
        col2.metric("Firm Value", f"${results['firm_value']:,.0f}")
        col3.metric("Terminal Value", f"${results['terminal_value']:,.0f}")
        col4.metric("Market Value of Debt", f"${results['market_value_of_debt']:,.0f}")
        
        st.subheader("Cost of Capital (WACC)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WACC", f"{results['WACC']*100:.2f}%")
        c2.metric("Cost of Equity", f"{results['cost_of_equity']*100:.2f}%")
        c3.metric("Cost of Debt (After Tax)", f"{results['cost_of_debt']*100:.2f}%")
        c4.metric("Unlevered Beta", f"{results['unlevered_beta']:.4f}")
        
        st.subheader("Capital Returns")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Invested Capital (Adj)", f"${results['invested_capital_adj']:,.0f}")
        c6.metric("Sum of PV(FCFF)", f"${results['sum_vp_FCFF']:,.0f}")
        
        st.subheader("Revenue & Free Cash Flow to Firm")
        chart_data = df.iloc[2:12][['Periodo', 'Ingresos', 'FCFF']].set_index('Periodo')
        st.line_chart(chart_data)
        
        try:
            excel_data = to_excel(df, results)
            st.download_button(
                label="Download Valuation to Excel",
                data=excel_data,
                file_name=f"Valuation_{company_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error generating Excel file. Need xlsxwriter module. Run: pip install xlsxwriter")

    with tab2:
        st.header("10-Year Projections (DataFrame)")
        st.markdown("Tabla con todos los flujos desde el Año 0 Base hasta el Año Terminal.")
        
        format_dict = {
            'Ingresos': '${:,.0f}',
            'Anual_growth_rate': '{:.2%}',
            'Operating_margin': '{:.2%}',
            'Operating_income': '${:,.0f}',
            'TAXES': '${:,.0f}',
            'Ebit(1-t)': '${:,.0f}',
            'StCR': '{:.2f}',
            'Reinvestment': '${:,.0f}',
            'FCFF': '${:,.0f}',
            'pv_FCFF': '${:,.0f}',
            'Inv_Capital': '${:,.0f}',
            'ROIC': '{:.2%}'
        }
        st.dataframe(df.style.format(format_dict), height=500)

    with tab3:
        st.header("Montecarlo Simulation")
        st.markdown("""
        ### ¿Cómo funciona la simulación?
        La simulación de Montecarlo ejecuta miles de escenarios variando aleatoriamente tres parámetros clave:
        1. **Margen EBIT**: Variación normal alrededor del margen operativo ingresado.
        2. **WACC**: Variación de la tasa de descuento utilizada para traer los flujos a valor presente.
        3. **Crecimiento a Perpetuidad (Terminal Growth)**: Variación alrededor de la Risk Free Rate.
        
        Estas combinaciones determinan distintas proyecciones de inversión y crecimiento, devolviendo un conjunto estadístico del 'Value per share'. Múltiples iteraciones filtran combinaciones no lógicas (e.g. WACC < Crecimiento).
        """)
        
        st.subheader("Configuración de la Simulación")
        c1, c2 = st.columns(2)
        n_sims = c1.slider("Number of Simulations", min_value=1000, max_value=20000, value=5000, step=1000)
        margen_ebit_std_pct = c2.number_input("Operating Margin Vol. (%)", value=2.00, format="%.2f")
        wacc_sim_std_pct = c1.number_input("WACC Volatility (%)", value=1.00, format="%.2f")
        term_growth_std_pct = c2.number_input("Terminal Growth Vol. (%)", value=0.50, format="%.2f")
        
        if st.button("Run Simulation"):
            with st.spinner("Running Monte Carlo Simulations..."):
                valores_por_accion = run_montecarlo_sim(
                    inputs, 
                    results['WACC'], 
                    n_simulaciones=n_sims,
                    margen_ebit_std=margen_ebit_std_pct / 100.0,
                    wacc_sim_std=wacc_sim_std_pct / 100.0,
                    term_growth_std=term_growth_std_pct / 100.0
                )
                
                if valores_por_accion:
                    media = np.mean(valores_por_accion)
                    mediana = np.median(valores_por_accion)
                    percentil_5 = np.percentile(valores_por_accion, 5)
                    percentil_95 = np.percentile(valores_por_accion, 95)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean Value per Share", f"${media:.2f}")
                    m2.metric("Median Value per Share", f"${mediana:.2f}")
                    m3.metric("90% Conf. Interval", f"${percentil_5:.2f} - ${percentil_95:.2f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(valores_por_accion, bins=50, edgecolor="black", alpha=0.7)
                    ax.axvline(media, color="red", linestyle="--", label=f"Mean: ${media:.2f}")
                    ax.axvline(percentil_5, color="green", linestyle="--", label=f"5th Pctl: ${percentil_5:.2f}")
                    ax.axvline(percentil_95, color="green", linestyle="--", label=f"95th Pctl: ${percentil_95:.2f}")
                    ax.set_xlabel("Value per Share ($)")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Fair Value per Share (Monte Carlo)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                else:
                    st.error("No valid values generated. Check inputs.")
                    
    with tab8:
        st.header("Detalle y Valor (Final Valuation Bridge)")
        st.markdown("Reconstrucción paso a paso del valor estimado por acción partiendo de los flujos de caja libre a la firma.")
        
        bridge_data = [
            {"Paso": "Terminal cash flow", "Fórmula": "FCFF del año terminal", "Valor": f"${df['FCFF'].iloc[12]:,.0f}"},
            {"Paso": "Terminal WACC", "Fórmula": "Costo de capital del año terminal", "Valor": f"{results['terminal_wacc']:.2%}"},
            {"Paso": "Terminal value", "Fórmula": "Terminal CF / (Terminal WACC - RFR)", "Valor": f"${results['terminal_value']:,.0f}"},
            {"Paso": "Present Value of Terminal value", "Fórmula": "Terminal Value / (1+WACC)^10", "Valor": f"${results['pv_terminal_value']:,.0f}"},
            {"Paso": "PV (CF over next 10 years)", "Fórmula": "Suma del PV de FCFF Años 1-10", "Valor": f"${results['pv_cf_1_10']:,.0f}"},
            {"Paso": "Value of operating assets", "Fórmula": "PV Terminal + PV (CF 1-10)", "Valor": f"${results['value_of_operating_assets']:,.0f}"},
            {"Paso": "Cash", "Fórmula": "+ Efectivo Año Base", "Valor": f"${inputs['cash_base_year']:,.0f}"},
            {"Paso": "Debt", "Fórmula": "- Deuda Año Base", "Valor": f"-${inputs['debt_base_year']:,.0f}"},
            {"Paso": "Minority interests", "Fórmula": "- Intereses Minoritarios", "Valor": f"-${inputs['minority_interes']:,.0f}"},
            {"Paso": "Non-operating assets", "Fórmula": "+ Activos No Operativos", "Valor": f"${inputs['non_operating_assets']:,.0f}"},
            {"Paso": "Value of Equity", "Fórmula": "= Valor total del Equity", "Valor": f"${results['value_of_equity']:,.0f}"},
            {"Paso": "Value of options", "Fórmula": "- Black-Scholes Value of Employee Options", "Valor": f"-${results['value_option']:,.0f}"},
            {"Paso": "Value of Equity in Common Stock", "Fórmula": "= Equidad en Acciones Comunes", "Valor": f"${results['value_in_common_stock']:,.0f}"},
            {"Paso": "Number of outstanding shares", "Fórmula": "Acciones en Circulación", "Valor": f"{inputs['shares_outstanding']:,.2f}"},
            {"Paso": "Estimated value / share", "Fórmula": "Value of Equity in Common Stock / Shares", "Valor": f"${results['value_per_share']:,.2f}"},
            {"Paso": "Current Price of the stock", "Fórmula": "Precio actual de mercado", "Valor": f"${inputs['current_share_price']:,.2f}"}
        ]
        
        bridge_df = pd.DataFrame(bridge_data)
        st.table(bridge_df.style.set_properties(**{'text-align': 'left'}))

    with tab4:
        st.header("WACC Calculation Details")
        st.markdown("Basado en el módulo `wacc.py`.")
        wacc_df = pd.DataFrame({
            "Componente": ["Unlevered Beta", "Cost of Equity", "Pre-tax Cost of Debt", "Cost of Debt (After Tax)", "Market Value of Debt", "Weight of Equity", "Weight of Debt", "WACC"],
            "Valor": [f"{results['unlevered_beta']:.4f}", f"{results['cost_of_equity']:.2%}", f"N/A", f"{results['cost_of_debt']:.2%}", f"${results['market_value_of_debt']:,.0f}", f"{results['weight_equity']:.2%}", f"{results['weight_debt']:.2%}", f"{results['WACC']:.2%}"]
        })
        st.table(wacc_df)

    with tab5:
        st.header("Sales to Capital Details")
        st.markdown("Basado en el módulo `sales_to_capital.py`.")
        stcr_df = pd.DataFrame({
            "Métrica": ["Capital Invertido Base", "Capital Invertido Ajustado (con R&D)", "Sales to Capital Ratio (Base)", "Sales to Capital Ratio (Ajustado)"],
            "Valor": [f"${results['invested_capital_base']:,.0f}", f"${results['invested_capital_adj']:,.0f}", f"{results['sales_to_capital_ratio_base']:.2f}", f"{results['sales_to_capital_ratio_adj']:.2f}"]
        })
        st.table(stcr_df)

    with tab6:
        st.header("R&D Adjustment Details")
        st.markdown("Basado en el módulo `rd_adjustment.py`. Muestra cómo se capitalizan los gastos en I+D (Research & Development).")
        rd_df = pd.DataFrame({
            "Ajuste": ["Capital Adjustment (Suma al Capital Invertido)", "R&D Current Amortization", "Income Adjustment (Suma al Operating Income)", "Tax Adjustment"],
            "Valor": [f"${results['rd_capital_adjustment']:,.0f}", f"${results['rd_amortization']:,.0f}", f"${results['rd_income_adjust']:,.0f}", f"${results['rd_tax_adjust']:,.0f}"]
        })
        st.table(rd_df)

    with tab7:
        st.header("Value Options Details (Black-Scholes-Merton)")
        st.markdown("Cálculo del efecto de dilución por opciones para empleados utilizando teória BSM.")
        
        d1_val = results['d1'] if results['d1'] != 0 else "N/A"
        d2_val = results['d2'] if results['d2'] != 0 else "N/A"
        call_val = f"${results['call_price']:,.2f}" if results['call_price'] != 0 else "N/A"
        
        opt_df = pd.DataFrame({
            "Variable": [
                "Asset Price (S)", 
                "Strike Price (K)", 
                "Maturity (t)", 
                "Volatility (σ)",
                "BSM d1",
                "BSM d2",
                "Call Option Price",
                "Total Options (Shares)",
                "Employee Options Total Value (Restado al Equity)"
            ],
            "Valor": [
                f"${inputs['current_share_price']:.2f}",
                f"${inputs.get('strike_price', 0):.2f}",
                f"{inputs.get('option_maturity', 0):.2f} years",
                f"{inputs.get('stock_volatility', 0):.2%}",
                f"{d1_val:.4f}" if isinstance(d1_val, float) else d1_val,
                f"{d2_val:.4f}" if isinstance(d2_val, float) else d2_val,
                call_val,
                f"{inputs.get('option_shares', 0):,.0f}",
                f"${results['value_option']:,.0f}"
            ]
        })
        st.table(opt_df.style.set_properties(**{'text-align': 'left'}))
                        
else:
    st.info("Adjust the values on the sidebar and click 'Run Valuation' to see results.")
