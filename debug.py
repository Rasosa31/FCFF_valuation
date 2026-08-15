import json
from valuation_engine import calculate_valuation
from datetime import datetime

with open('/Users/ramirososa/Documents/Valuations/NVDA/Inputs_Save-7.json', 'r') as f:
    saved_data = json.load(f)

# Mock app.py session state extraction
inputs = {
    'revenue_base_year': saved_data['rev_base'],
    'cash_base_year': saved_data['cash_base'],
    'equity_base_year': saved_data['eq_base'],
    'debt_base_year': saved_data['debt_base'],
    'interes_expenses': saved_data['int_exp'],
    'income_base_year': saved_data['inc_base'],
    'minority_interes': saved_data['min_int'],
    'non_operating_assets': saved_data['non_op'],
    
    'marginal_tax_rate': saved_data['mar_tax'],
    
    'shares_outstanding': saved_data['shares'],
    'current_share_price': saved_data['price'],
    'beta_option': saved_data['beta_opt_single'],
    'unlevered_beta': saved_data['unlev_beta_cash'],
    'ERP': saved_data['erp'],
    'RFR': saved_data['rfr'],
    
    'av_maturity_of_debt': saved_data['mat_debt'],
    'base_r_d_expenses': saved_data.get('rd_curr', 0),
    'minus_oneyear_r_d_expense': saved_data.get('rd_m1', 0),
    'minus_twoyear_r_d_expense': saved_data.get('rd_m2', 0),
    'minus_threeyear_r_d_expense': saved_data.get('rd_m3', 0),
    
    'agr_rate': [0.0]*10,
    'op_margin': [0.0]*10,
    'et_rate': [0.0]*10,
    'stcr_projection': [2.5]*10,
    'terminal_operating_margin': 0.1,
    'terminal_reinvestment_method': "Terminal ROIC (Damodaran Base)",
    'terminal_stcr_input': "",
    
    'options_calc_method': "Usar Black-Scholes",
    'option_shares': saved_data.get('opt_shares', 0),
    'strike_price': saved_data.get('strike', 0),
    'option_maturity': saved_data.get('opt_mat', 0),
    'stock_volatility': saved_data.get('volatility', 0),
    'manual_options_value': saved_data.get('manual_opt_val', 0)
}

try:
    df, results = calculate_valuation(inputs)
    mc = inputs['current_share_price'] * inputs['shares_outstanding']
    
    if inputs['beta_option'] == "Sectorial Normal":
        debt_for_beta = inputs['debt_base_year']
    else:
        debt_for_beta = max(0, inputs['debt_base_year'] - inputs['cash_base_year'])
        
    e_beta = mc if mc > 0 else inputs['equity_base_year']
    
    print(f"Inputs:")
    print(f"price = {inputs['current_share_price']}")
    print(f"shares = {inputs['shares_outstanding']}")
    print(f"eq_base = {inputs['equity_base_year']}")
    print(f"mc = {mc}")
    print(f"d_beta = {debt_for_beta}")
    print(f"e_beta = {e_beta}")
    print(f"Beta Option = {inputs['beta_option']}")
    print(f"results['levered_beta'] = {results['levered_beta']}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
