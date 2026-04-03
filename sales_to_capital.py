from rd_adjustment import result, result1
from inputs import (
    revenue_base_year,
    equity_base_year,
    income_base_year,
    cash_base_year,
    debt_base_year,
)


class Stcr:
    def __init__(
        self,
        revenue_base_year,
        equity_base_year,
        income_base_year,
        cash_base_year,
        debt_base_year,
    ):
        self.revenue_base_year = revenue_base_year
        self.equity_base_year = equity_base_year
        self.income_base_year = income_base_year
        self.cash_base_year = cash_base_year
        self.debt_base_year = debt_base_year

    def invested_capital(self):
        return round(
            (
                self.equity_base_year
                + self.debt_base_year
                - self.cash_base_year
                + result
            ),
            2,
        )

    def sales_to_capital_ratio(self):
        return round((self.revenue_base_year) / (self.invested_capital()), 2)

    def ROIC(self):
        return round(self.income_base_year / self.invested_capital(), 4)


# revenue_base_year = float(input(f"ingrese el valor de los ingresos o ventas del año base:  "))
# equity_base_year = float(input(f"ingrese el valor del equity en el año base:  "))
# debt_base_year = float(input(f"ingrese el valor de la deuda en el año base:  "))
# cash_base_year = float(input(f"ingrese el valor del efectivo en el año base:  "))
# income_base_year = float(input(f"ingrese el valor del income en el año base:  "))

stcr = Stcr(
    revenue_base_year,
    equity_base_year,
    income_base_year,
    cash_base_year,
    debt_base_year,
)

print(f"Final invested capital: {stcr.invested_capital()}")
print(f"Sales To Capital Ratio: {stcr.sales_to_capital_ratio()}")
print(f"Return On Invested Capital INICIAL: {stcr.ROIC()}")

# Guardar en archivo .txt
with open("valoracion_ARM.txt", "w") as file:
    file.write(f" Final invested capital: {stcr.invested_capital()}")
    file.write(f" Sales To Capital Ratio: {stcr.sales_to_capital_ratio()}")
    file.write(f"Return On Invested Capital INICIAL: {stcr.ROIC()}")

print("Resultados guardados en valoracion_ARM.txt")
