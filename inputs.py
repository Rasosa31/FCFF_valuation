# Base data
revenue_base_year = 175294
cash_base_year = 38066
equity_base_year = 82324
debt_base_year = 31454
interes_expenses = 1258
income_base_year = 17186
# FCFF - 10
agr_rate = 0.05  # Anual Growth revwnue growth
op_margin = (
    0.10  # Resultado de dividir el Operating Income entre las ventas o ingresos.
)
et_rate = 0.14  # Efective tax rate
marginal_tax_rate = 0.25
# reinvestment_rate = 0.208247840 # Este calculo se hizo en este ejemplo sobre las ventas o ingresos
# Terminal year
RFR = 0.0445
terminal_growth_rate = RFR
terminal_operating_margin = 0.10
terminal_tax_rate = marginal_tax_rate
# terminal_reinvestment_rate = 0.06

shares_outstanding = 2941.6
current_share_price = 12.7
levered_beta = 1.24

ERP = 0.0429

base_r_d_expenses = 5392
minus_oneyear_r_d_expense = 5493
minus_twoyear_r_d_expense = 5122
minus_threeyear_r_d_expense = 4336

av_maturity_of_debt = 5  # Promedio de edad de la deuda total de la compañia
minority_interes = 599
non_operating_assets = 0

print(f"Minority Interes: {minority_interes}")
print(f"non operating assets: {non_operating_assets}")
