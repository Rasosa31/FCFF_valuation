import pandas as pd
import numpy as np

def calculate_valuation(inputs):
    """
    Computes FCFF, WACC and other valuation metrics based on inputs.
    Returns a pandas DataFrame with projections and a dictionary with key metrics.
    """
    
    # 1. WACC & Cost of Capital Calculation
    mc = inputs['current_share_price'] * inputs['shares_outstanding']
    
    unlevered_beta = inputs['levered_beta'] / (
        1 + (1 - inputs['marginal_tax_rate']) * (inputs['debt_base_year'] / mc)
    )
    
    cost_of_equity = inputs['RFR'] + (unlevered_beta * inputs['ERP'])
    
    pretax_cost_of_debt = inputs['interes_expenses'] / inputs['debt_base_year'] if inputs['debt_base_year'] > 0 else 0
    cost_of_debt = pretax_cost_of_debt * (1 - inputs['marginal_tax_rate'])
    
    market_value_of_debt = inputs['debt_base_year'] / (1 + cost_of_debt) ** inputs['av_maturity_of_debt'] if cost_of_debt > 0 else inputs['debt_base_year']
    
    weight_equity = mc / (mc + market_value_of_debt)
    weight_debt = market_value_of_debt / (mc + market_value_of_debt)
    
    wacc = (cost_of_equity * weight_equity) + (cost_of_debt * weight_debt)
           
    # 2. R&D Adjustment
    rd_capital_adjustment = (inputs['base_r_d_expenses']) + \
                            (inputs['minus_oneyear_r_d_expense'] * 0.75) + \
                            (inputs['minus_twoyear_r_d_expense'] * 0.50)
                            
    rd_amortization = (inputs['minus_oneyear_r_d_expense'] * 0.333) + \
                      (inputs['minus_twoyear_r_d_expense'] * 0.333) + \
                      (inputs['minus_threeyear_r_d_expense'] * 0.333)
                      
    rd_income_adjust = inputs['base_r_d_expenses'] - rd_amortization
    rd_tax_adjust = rd_income_adjust * inputs['marginal_tax_rate']
    
    # 3. Invested Capital & Sales to Capital Ratio
    invested_capital_base = inputs['equity_base_year'] + inputs['debt_base_year'] - inputs['cash_base_year']
    invested_capital_adj = round((invested_capital_base + rd_capital_adjustment), 2)
    
    sales_to_capital_ratio_base = round((inputs['revenue_base_year']) / invested_capital_base, 2) if invested_capital_base != 0 else 0
    sales_to_capital_ratio_adj = round((inputs['revenue_base_year']) / invested_capital_adj, 2) if invested_capital_adj != 0 else 0
    
    roic_base = round(inputs['income_base_year'] / invested_capital_base, 4) if invested_capital_base != 0 else 0
    roic_adj = round((inputs['income_base_year'] + rd_income_adjust - rd_tax_adjust) / invested_capital_adj, 4) if invested_capital_adj != 0 else 0

    # 4. Projections & FCFF DataFrame (13 Rows: 0, 0A, 1-10, Terminal)
    labels = ["Año 0", "Año 0A"] + [f"Año {i}" for i in range(1, 11)] + ["Terminal"]
    
    # Expand scalar inputs to lists if needed
    agr_rate_list = inputs['agr_rate'] if isinstance(inputs['agr_rate'], list) else [inputs['agr_rate']] * 10
    op_margin_list = inputs['op_margin'] if isinstance(inputs['op_margin'], list) else [inputs['op_margin']] * 10
    et_rate_list = inputs['et_rate'] if isinstance(inputs['et_rate'], list) else [inputs['et_rate']] * 10
    stcr_list = inputs.get('stcr_projection', [0.8]*10)
    stcr_list = stcr_list if isinstance(stcr_list, list) else [stcr_list] * 10
    
    # Revenue projections
    ingresos_base = inputs['revenue_base_year']
    ingresos = [ingresos_base, ingresos_base] # Year 0 and 0A have same revenue
    
    ingresos.append(ingresos_base) # Year 1
    
    for i in range(1, 10):
        tasa_crecimiento = agr_rate_list[i]
        ingresos.append(ingresos[-1] * (1 + tasa_crecimiento))
        
    ingreso_term_year = ingresos[-1] * (1 + inputs['RFR'])
    ingresos.append(ingreso_term_year)
        
    df = pd.DataFrame({"Periodo": labels, "Ingresos": ingresos})
    
    # Anual_growth_rate
    agr_visual = [0, 0, 0] + agr_rate_list[1:] + [inputs['RFR']]
    df["Anual_growth_rate"] = agr_visual
    
    # Margins & EBIT (Operating_income)
    ebit = [
        inputs['income_base_year'],                     # Año 0
        inputs['income_base_year'] + rd_income_adjust,  # Año 0A
        inputs['income_base_year'] + rd_income_adjust   # Año 1
    ]
    for i in range(1, 10):
        ebit.append(ingresos[i+2] * op_margin_list[i]) # Año 2-10
    ebit.append(ingresos[-1] * inputs['terminal_operating_margin']) # Terminal
    
    margen_operacional = [0, 0] + [ebit[2] / ingresos[2] if ingresos[2] != 0 else 0] + op_margin_list[1:] + [inputs['terminal_operating_margin']]
    
    df["Operating_margin"] = margen_operacional
    df["Operating_income"] = ebit
    
    # Taxes
    et_rate_base = et_rate_list[0]
    taxes = [
        inputs['income_base_year'] * et_rate_base, # Año 0
        (inputs['income_base_year'] + rd_income_adjust) * et_rate_base, # Año 0A
        (inputs['income_base_year'] + rd_income_adjust) * et_rate_base  # Año 1
    ]
    for i in range(3, 12):
        taxes.append(ebit[i] * et_rate_list[i-2]) # Año 2-10
    taxes.append(ebit[12] * inputs['marginal_tax_rate']) # Terminal
    df["TAXES"] = taxes
    
    ebit_less_taxes = [ebit[i] - taxes[i] for i in range(13)]
    df["Ebit(1-t)"] = ebit_less_taxes
    
    # Reinvestment & StCR
    sales_to_capital = [sales_to_capital_ratio_base, sales_to_capital_ratio_adj]
    actual_stcr_list = [sales_to_capital_ratio_adj] + stcr_list[1:]
    sales_to_capital.extend(actual_stcr_list)
    sales_to_capital.append(0) # Terminal
    
    reinvestment = [0, 0] # Year 0 and 0A
    for i in range(2, 12):
        reinvestment.append((ingresos[i] - ingresos[i-1]) / sales_to_capital[i])
    
    roic_10 = ebit_less_taxes[11] / (invested_capital_adj + sum(reinvestment[2:12])) if (invested_capital_adj + sum(reinvestment[2:12])) != 0 else 0
    frat = inputs['RFR'] / roic_10 if roic_10 != 0 else 0
    terminal_reinvestment = frat * ebit_less_taxes[12]
    reinvestment.append(terminal_reinvestment)
    
    df["StCR"] = sales_to_capital
    df["Reinvestment"] = reinvestment
    
    # FCFF
    FCFF = [0, 0] # Year 0 and 0A
    for i in range(2, 13):
        FCFF.append(ebit_less_taxes[i] - reinvestment[i])
    df["FCFF"] = FCFF
    
    # PV of FCFF
    pv_FCFF = [0, 0] + [FCFF[i] / (1 + wacc) ** (i-1) for i in range(2, 12)] + [0] # Terminal PV is calculated separately
    df["pv_FCFF"] = pv_FCFF
    
    # Cumulated Invested Capital and ROIC
    cumulated_inv_cap = [invested_capital_base, invested_capital_adj]
    current_inv_cap = invested_capital_adj
    for i in range(2, 13):
        if i < 12:
            current_inv_cap += reinvestment[i]
            cumulated_inv_cap.append(current_inv_cap)
        else:
            cumulated_inv_cap.append(cumulated_inv_cap[-1] + terminal_reinvestment)
            
    df["Inv_Capital"] = cumulated_inv_cap
    
    # ROIC calculation for each year
    roic = [roic_base, roic_adj] + [ebit_less_taxes[i] / cumulated_inv_cap[i-1] for i in range(2, 13)]
    df["ROIC"] = roic
    
    sum_vp_FCFF = sum(pv_FCFF[2:12]) # Only sum years 1 to 10
    
    from scipy.stats import norm

    # 5. Terminal Value Calculation
    FCFF_terminal = FCFF[12]
    term_wacc = inputs.get('terminal_wacc', wacc) # Fallback to WACC if not provided
    
    term_discount = term_wacc - inputs['RFR']
    if term_discount <= 0.0001:
        term_discount = 0.0001
        
    terminal_value = FCFF_terminal / term_discount
    valor_presente_terminal_value = terminal_value / (1 + wacc) ** 10 # Tráido a valor presente usando WACC current de 10 años
    
    # PV (CF over next 10 years)
    pv_cf_1_10 = sum_vp_FCFF
    
    # Value of operating assets
    value_of_operating_assets = valor_presente_terminal_value + pv_cf_1_10
    
    # Value of equity
    value_of_equity = value_of_operating_assets + inputs['cash_base_year'] - inputs['debt_base_year'] - inputs['minority_interes'] + inputs['non_operating_assets']
    
    # Value Options (Black-Scholes-Merton)
    option_shares = inputs.get('option_shares', 0)
    d1, d2, call_price = 0, 0, 0
    if option_shares > 0:
        S = inputs['current_share_price']
        K = inputs.get('strike_price', 0)
        t = inputs.get('option_maturity', 0)
        r = inputs['RFR']
        sigma = inputs.get('stock_volatility', 0)
        
        if K > 0 and t > 0 and sigma > 0:
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            # Call option price
            call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
            value_option = call_price * option_shares
        else:
            value_option = max(0, (S - K) * option_shares) # Intrinsic fallback
            call_price = max(0, S - K)
    else:
        value_option = 0
            
    # Value of equity in common stock
    value_in_common_stock = value_of_equity - value_option
    
    # Estimated value / share
    value_per_share = value_in_common_stock / inputs['shares_outstanding'] if inputs['shares_outstanding'] > 0 else 0
    
    # For backward compatibility with older dictionary keys if needed in other tabs
    firm_value = value_of_equity + inputs['debt_base_year'] # Firm value is operating + cash + non-ops, or Equity + Debt
    
    # Dictionary packing all detailed items for the tabs
    results = {
        'WACC': wacc,
        'terminal_wacc': term_wacc,
        'terminal_value': terminal_value,
        'pv_terminal_value': valor_presente_terminal_value,
        'pv_cf_1_10': pv_cf_1_10,
        'sum_vp_FCFF': sum_vp_FCFF,
        'value_of_operating_assets': value_of_operating_assets,
        'value_of_equity': value_of_equity,
        'value_in_common_stock': value_in_common_stock,
        'firm_value': firm_value,
        'value_per_share': value_per_share,
        'invested_capital_base': invested_capital_base,
        'invested_capital_adj': invested_capital_adj,
        'sales_to_capital_ratio_base': sales_to_capital_ratio_base,
        'sales_to_capital_ratio_adj': sales_to_capital_ratio_adj,
        'unlevered_beta': unlevered_beta,
        'cost_of_equity': cost_of_equity,
        'cost_of_debt': cost_of_debt,
        'market_value_of_debt': market_value_of_debt,
        'weight_equity': weight_equity,
        'weight_debt': weight_debt,
        'rd_capital_adjustment': rd_capital_adjustment,
        'rd_amortization': rd_amortization,
        'rd_income_adjust': rd_income_adjust,
        'rd_tax_adjust': rd_tax_adjust,
        'value_option': value_option,
        # BSM details
        'd1': d1,
        'd2': d2,
        'call_price': call_price
    }
    
    return df, results

def run_montecarlo_sim(inputs, wacc_base, n_simulaciones=5000, margen_ebit_std=0.02, wacc_sim_std=0.01, term_growth_std=0.005):
    np.random.seed(42)
    anios = 10
    ingresos_iniciales = inputs['revenue_base_year']
    deuda_neta = inputs['debt_base_year'] - inputs['cash_base_year'] + inputs['minority_interes'] - inputs['non_operating_assets']
    acciones = inputs['shares_outstanding']
    
    agr_rate_list = inputs['agr_rate'] if isinstance(inputs['agr_rate'], list) else [inputs['agr_rate']] * 10
    op_margin_list = inputs['op_margin'] if isinstance(inputs['op_margin'], list) else [inputs['op_margin']] * 10
    et_rate_list = inputs['et_rate'] if isinstance(inputs['et_rate'], list) else [inputs['et_rate']] * 10
    stcr_list = inputs.get('stcr_projection', [0.8]*10)
    stcr_list = stcr_list if isinstance(stcr_list, list) else [stcr_list] * 10

    # Deterministic Revenues (matches base valuation)
    ingresos = [ingresos_iniciales] # Year 0
    ingresos.append(ingresos_iniciales) # Year 1
    for i in range(1, 10):
        tasa_crecimiento = agr_rate_list[i]
        ingresos.append(ingresos[-1] * (1 + tasa_crecimiento))
    
    # Distributions
    margin_shock = np.random.normal(0, margen_ebit_std, n_simulaciones)
    wacc_sim = np.random.normal(wacc_base, wacc_sim_std, n_simulaciones)
    crecimiento_perpetuo = np.random.normal(inputs['RFR'], term_growth_std, n_simulaciones)
    
    valores_por_accion = []
    
    for i in range(n_simulaciones):
        fcff_proyectados = []
        for t in range(1, anios + 1):
            ebit = ingresos[t] * (op_margin_list[t-1] + margin_shock[i])
            ebit_1_t = ebit * (1 - et_rate_list[t-1])
            delta_sales = ingresos[t] - ingresos[t-1]
            reinvestment = delta_sales / stcr_list[t-1]
            fcff = ebit_1_t - reinvestment
            fcff_proyectados.append(fcff)
            
        w_sim = wacc_sim[i]
        g_sim = crecimiento_perpetuo[i]
        
        if w_sim <= g_sim or w_sim <= 0:
            continue
            
        valor_empresa = 0
        for t in range(1, anios + 1):
            valor_empresa += fcff_proyectados[t - 1] / (1 + w_sim) ** t
            
        valor_terminal = (fcff_proyectados[-1] * (1 + g_sim)) / (w_sim - g_sim)
        valor_empresa += valor_terminal / (1 + w_sim) ** anios
        
        if valor_empresa <= deuda_neta:
            continue
            
        valor_equidad = valor_empresa - deuda_neta
        precio_accion = valor_equidad / acciones
        
        if precio_accion > 0:
            valores_por_accion.append(precio_accion)
            
    return valores_por_accion
