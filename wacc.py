from os import write

from inputs import (
    shares_outstanding,
    current_share_price,
    levered_beta,
    RFR,
    ERP,
    debt_base_year,
    interes_expenses,
    marginal_tax_rate,
    equity_base_year,
    av_maturity_of_debt,
)


class CostOfCapital:
    def __init__(
        self,
        shares_outstanding,
        current_share_price,
        levered_beta,
        RFR,
        ERP,
        debt_base_year,
        interes_expenses,
        marginal_tax_rate,
        equity_base_year,
        av_maturity_of_debt,
    ):
        self.shares_outstanding = shares_outstanding
        self.current_share_price = current_share_price
        self.levered_beta = levered_beta
        self.RFR = RFR
        self.ERP = ERP
        self.debt_base_year = debt_base_year
        self.interes_expenses = interes_expenses
        self.marginal_tax_rate = marginal_tax_rate
        self.equity_base_year = equity_base_year
        self.av_maturity_of_debt = av_maturity_of_debt

    def market_capitalization(self):
        return self.current_share_price * self.shares_outstanding

    def unlevered_beta(self):
        return self.levered_beta / (
            1
            + (1 - self.marginal_tax_rate)
            * (self.debt_base_year / self.market_capitalization())
        )

    def cost_of_equity(self):
        return self.RFR + (self.unlevered_beta() * self.ERP)

    def pretax_cost_of_debt(self):
        return self.interes_expenses / self.debt_base_year

    def cost_of_debt(self):
        return self.pretax_cost_of_debt() * (1 - self.marginal_tax_rate)

    def market_value_of_debt(self):
        return (
            self.debt_base_year / (1 + self.cost_of_debt()) ** self.av_maturity_of_debt
        )

    def cost_of_equity_weighted(self):
        return self.cost_of_equity() * (
            self.market_capitalization()
            / (self.market_capitalization() + self.market_value_of_debt())
        )

    def cost_of_debt_weighted(self):
        return (
            self.cost_of_debt()
            * self.market_value_of_debt()
            / (self.market_capitalization() + self.market_value_of_debt())
        )

    def wacc(self):
        return self.cost_of_equity_weighted() + self.cost_of_debt_weighted()


# Solicitar al usuario los valores mediante input()

# shares_outstanding = float(input(f"ingrese el numero de acciones en circulacion: "))
# current_share_price = float(input(f"ingrese precio actual de la accion: "))
# levered_beta = float(input(f"ingrese la beta apalancada: "))
# risk_free_rate = float(input(f"ingrese la tasa libre de riesgo: "))
# equity_risk_premium = float(input(f"ingrese la tasa Equity Risk Premium: "))
# book_value_debt = float(input(f"ingrese el valor de la deuda en libros: "))
# interes_expenses = float(input(f"ingrese los intereses pagados en el año: "))
# marginal_tax_rate = float(input(f"ingrese la tasa marginal del impuesto de renta: "))
# equity = float(input(f"ingrese el book value del equity: "))
# av_maturity_of_debt = float(input(f"ingrese el numero promedio años del vencimiento de la deuda: "))

# Crear instancia de la clase con los valores proporcionados por el usuario
wacc_fcff = CostOfCapital(
    shares_outstanding,
    current_share_price,
    levered_beta,
    RFR,
    ERP,
    debt_base_year,
    interes_expenses,
    marginal_tax_rate,
    equity_base_year,
    av_maturity_of_debt,
)

print(f"Market Capitalization: {wacc_fcff.market_capitalization()}")
print(f"Unlevered Beta: {wacc_fcff.unlevered_beta():0.04f}")
print(f"Cost of Equity: {wacc_fcff.cost_of_equity():0.4f}")
print(f"Pre-tax cost of debt: {wacc_fcff.pretax_cost_of_debt():0.4f}")
print(f"Cost of Debt: {wacc_fcff.cost_of_debt():0.4f}")
print(f"Market Value Of Debt: {wacc_fcff.market_value_of_debt():0.0f}")
print(f"Cost Of Equity ponderado: {wacc_fcff.cost_of_equity_weighted():0.4f}")
print(f"Cost of Debt Ponderado: {wacc_fcff.cost_of_debt_weighted():0.4f}")
print(f"COST OF DE CAPITAL(WACC): {wacc_fcff.wacc():0.4f}")


# Guardar en archivo .txt
with open("valoracion_ARM.txt", "w") as file:
    file.write(f"Cost of Equity: {wacc_fcff.cost_of_equity():0.4f}")
    file.write(f"Pre-tax cost of debt: {wacc_fcff.pretax_cost_of_debt():0.4f}")
    file.write(f"Cost of Debt: {wacc_fcff.cost_of_debt():0.4f}")
    file.write(f"Market Value Of Debt: {wacc_fcff.market_value_of_debt():0.0f}")
    file.write(f"Market Capitalization: {wacc_fcff.market_capitalization()}")
    file.write(f"Cost Of Equity ponderado: {wacc_fcff.cost_of_equity_weighted():0.4f}")
    file.write(f"Cost of Debt Ponderado: {wacc_fcff.cost_of_debt_weighted():0.4f}")
    file.write(f"COST OF DE CAPITAL(WACC): {wacc_fcff.wacc():0.6f}")


print("Resultados guardados en valoracion_ARM.txt")
