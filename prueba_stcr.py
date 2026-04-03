from inputs import (
    marginal_tax_rate,
    RFR,
    revenue_base_year,
    cash_base_year,
    debt_base_year,
    minority_interes,
    non_operating_assets,
    shares_outstanding,
    terminal_operating_margin,
)
from wacc import wacc_fcff
from sales_to_capital import stcr
from value_options import value_option
import pandas as pd  # para calculos matematicos y DataFrame
import numpy as np  # Para calculos matematicos y DataFrame

# Ingresos del año base
ingresos_base = revenue_base_year

# Diccionario con las tasas de crecimiento año a año
tasas_crecimiento = {
    1: 0.01,
    2: 0.01,
    3: 0.01,
    4: 0.02,
    5: 0.02,
    6: 0.02,
    7: 0.02,
    8: 0.02,
    9: 0.02,
    10: 0.02,
}

# Crear una lista para almacenar los ingresos de cada año
ingresos = [ingresos_base]

# Calcular los ingresos para cada año
for año in range(1, 11):
    ingresos_anterior = ingresos[-1]
    tasa_crecimiento = tasas_crecimiento[año]
    ingresos_actual = ingresos_anterior * (1 + tasa_crecimiento)
    ingresos.append(ingresos_actual)

# Crear un DataFrame con los resultados
df = pd.DataFrame({"Año": range(0, 11), "Ingresos": ingresos})

margen_operacional = {
    0: 0,
    1: 0.09,
    2: 0.08,
    3: 0.08,
    4: 0.07,
    5: 0.08,
    6: 0.08,
    7: 0.08,
    8: 0.08,
    9: 0.08,
    10: 0.09,
}

# Calcular el EBIT multiplicando los ingresos por el margen operacional
ebit = [ingresos[i] * margen_operacional[i] for i in range(len(ingresos))]

# Añadir el EBIT al DataFrame
df["Marg_Oper"] = [margen_operacional[i] for i in range(len(ingresos))]
df["EBIT"] = ebit

efective_tax_rate = {
    0: 0,
    1: 0.155,
    2: 0.155,
    3: 0.155,
    4: 0.155,
    5: 0.155,
    6: 0.155,
    7: 0.155,
    8: 0.155,
    9: 0.155,
    10: 0.155,
}

taxes = [ebit[i] * efective_tax_rate[i] for i in range(len(ebit))]

# Añadir el ebit_after_tax al DataFrame
# df['Ef_tax_rate'] = [efective_tax_rate[i] for i in range(len(ebit))]
df["TAXES"] = taxes

# ebit MENOS IMPUESTOS EBIT(1-t)
ebit_less_taxes = [ebit[i] - taxes[i] for i in range(len(ebit))]
df["Ebit(1-t)"] = ebit_less_taxes

sales_to_capital = {
    0: stcr.sales_to_capital_ratio(),
    1: 0.8,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
}  # El año 11 o terminal esta en cero, se calcula abajo.

reinvestment = [
    (ingresos[i] - ingresos[i - 1]) / sales_to_capital[i] if i > 0 else 0
    for i in range(len(ingresos))
]

df["StCR"] = [sales_to_capital[i] for i in range(len(ingresos))]
df["Reinvestment"] = reinvestment

# Calculo del FCFF para cada año de la serie

FCFF = [(ebit[i] - taxes[i]) - reinvestment[i] for i in range(len(ebit))]
df["FCFF"] = [FCFF[i] for i in range(len(ebit))]
df["FCFF"] = FCFF  # Linea para incluir el resultado en el DatFrame

# Calculo que devuelve cada año de la serie a valor presente
pv_FCFF = [fcff / (1 + wacc_fcff.wacc()) ** n for n, fcff in enumerate(FCFF, start=0)]
df["pv_FCFF"] = pv_FCFF  # Línea que incluye los resultados al DataFrame

# Seccion para calcular el ROIC de la serie del FCFF a 10 años

inv_capital = (
    stcr.invested_capital()
)  # importacion del valor del capital invertido en el año cero desde otro archivo

# Se crea una lista vacia para almacenar los valores anuales del capital invertido acumulado
cumulated_inv_cap = []

# Itera sobre cada año y suma el reinvestment al capital invertido
for año in range(len(reinvestment)):
    inv_capital += reinvestment[año]
    cumulated_inv_cap.append(inv_capital)

df["Inv Capital"] = cumulated_inv_cap  # Línea para incluir el resultado en el DataFrame

roic = [
    (ebit_less_taxes[i] / cumulated_inv_cap[i]) for i in range(len(ebit))
]  # Lí que hace la iteracion.
df["ROIC"] = roic

sum_vp_FCFF = sum(
    pv_FCFF
)  # Operación para sumar los valores presentes del FCFF de la serie

# Factor para calcular la reinversion del año terminal
# Según la metodología propuesta el valor de la reinversión en el año terminal se calcula multiplicando el ....
# ... resultado de la división del RFR entre el ROIC del año 10 por el ebit(1-t) del año terminal. Damodaran
roic_10 = roic[10]  # forma de llamar el valor del ROIC en el año 10
frat = (
    RFR / roic_10
)  # factor de reinversion para calcularla en el año terminal. multiplica ebit del año terminal


# CALCULO DEL TERMINAL VALUE FCFF
ingreso_term_year = ingresos[10] * (
    1 + RFR
)  # ingresos en el año terminal. Ingreso año por 1 + la tasa libre de riesgo
ebit_terminal_year = ingreso_term_year * terminal_operating_margin
tax_terminal_year = (
    ebit_terminal_year * marginal_tax_rate
)  # EBIT (1-t) del año terminal
terminal_reinvestment = frat * (
    ebit_terminal_year - tax_terminal_year
)  # Valor de la reinversion en el año terminal
FCFF_terminal = ebit_terminal_year - tax_terminal_year - terminal_reinvestment
terminal_value = (
    FCFF_terminal / RFR
)  # Es el resultado de calcular la perpetuidad del FCFF_Terminal, el cual luego
# debe traerse a valor presente utilizando el WACC como tasa de descuento)

# VALOR PRESENTE DEL VALOR TERMINAL

valor_presente_terminal_value = terminal_value / (1 + wacc_fcff.wacc()) ** 10

firm_value = (
    valor_presente_terminal_value
    + sum_vp_FCFF
    + cash_base_year
    - debt_base_year
    - minority_interes
    - value_option
    + non_operating_assets
)

value_per_share = firm_value / shares_outstanding


# Mejorar la salida a consola
print(df.to_string(index=False))

print(f" Suma de los valores a valor presente del FCFF: ", sum_vp_FCFF)
print(f" reinvestment_año terminal: ", terminal_reinvestment)
print(f" Terminal_value: ", terminal_value)  # este valor debe traerse a valor presente.
print(f" Valor Presente del terminal_value: ", valor_presente_terminal_value)
print(f" EBIT(1-t): ", ebit_less_taxes[10])  # . confirmado
print(f" FCFF_terminal: ", FCFF_terminal)  # confirmado
print(f" Prueba: ", stcr)
print(f" prueba: ", wacc_fcff)
print(f" Valor de Compañia - Firm Value: ", firm_value)
print(f" Valor de la Acción: ", value_per_share)
print(f"ebit_less_taxes[10]: ", ebit_less_taxes[10])
print(f"ingreso terminal year: ", ingreso_term_year)  # OK
print(f"ebit_terminal_year: ", ebit_terminal_year)  # OK
print(f"tax_terminal_year: ", tax_terminal_year)  # OK
print(
    f"terminal_reinvestment: ", terminal_reinvestment
)  # (RFR / roic año 10) * EBIT(1-t) terminal year
print(f"FCFF_terminal: ", FCFF_terminal)  # OK
print(f"terminal_value: ", terminal_value)  # OK
print(f"Factor de reinversion año terminal: ", frat)
