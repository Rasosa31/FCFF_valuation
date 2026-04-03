from inputs import (
    base_r_d_expenses,
    minus_oneyear_r_d_expense,
    minus_twoyear_r_d_expense,
    minus_threeyear_r_d_expense,
    marginal_tax_rate,
)


class Rdajust:
    def __init__(
        self,
        base_r_d_expenses,
        minus_oneyear_r_d_expense,
        minus_twoyear_r_d_expense,
        minus_threeyear_r_d_expense,
        marginal_tax_rate,
    ):
        self.base_r_d_expenses = base_r_d_expenses
        self.minus_oneyear_r_d_expense = minus_oneyear_r_d_expense
        self.minus_twoyear_r_d_expense = minus_twoyear_r_d_expense
        self.minus_threeyear_r_d_expense = minus_threeyear_r_d_expense
        self.marginal_tax_rate = marginal_tax_rate

    # Este valor debe sumarse al capital invertido y así se obtiene el capital invertido ajustado
    # Procedimiento: La metodología incluye los últimos tres años reportados en el rubro R&D
    # Al último año reportado se suma la parte aún por amortizar de los 2 años anteriores
    # La suma de estos valores debe tenerse en cuenta para totalizar en el capital invertido
    def capital_adjustment(self):
        return (
            (self.base_r_d_expenses)
            + (self.minus_oneyear_r_d_expense * 0.75)
            + (self.minus_twoyear_r_d_expense * 0.50)
        )

    def amotization_current_year(self):
        return (
            (self.minus_oneyear_r_d_expense * 0.333)
            + (self.minus_twoyear_r_d_expense * 0.333)
            + (self.minus_threeyear_r_d_expense * 0.333)
        )

    def income_adjust(self):
        return (
            self.base_r_d_expenses - self.amotization_current_year()
        )  # este valor se ajusta en operating income
        # no en los ingresos iniciales

    def tax_adjust(self):
        return self.income_adjust() * self.marginal_tax_rate


# Crear instancia de la clase con los valores importados de 'inputs.py'
rdajust = Rdajust(
    base_r_d_expenses,
    minus_oneyear_r_d_expense,
    minus_twoyear_r_d_expense,
    minus_threeyear_r_d_expense,
    marginal_tax_rate,
)

# Resultados
result = rdajust.capital_adjustment()
print(f"Valor a ajustar en el Capital Invertido: {result: 0.2f}")
result1 = rdajust.income_adjust()
print(f"Ingreso a ajustar (sumar) al income: {result1: 0.2f}")
result2 = rdajust.tax_adjust()
print(f"Valor en impuestos a ajustar: {result2: 0.2f}")
result3 = rdajust.amotization_current_year()
print(f"valor de la amortizacion del presente año:  {result3}")

# Guardar en archivo .txt
with open("valoracion_ARM_RyD.txt", "w") as file:
    file.write(f"Valor a ajustar en el Capital Invertido: {result: 0.2f}\n")
    file.write(f"Ingreso a ajustar (sumar) al income: {result1: 0.2f}\n")
    file.write(f"Valor en impuestos a ajustar: {result2: 0.2f}\n")

print("Resultados guardados en valoracion_ARM_RyD.txt")
