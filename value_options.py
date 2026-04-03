def calculate_options_value(option_shares, stock_price, strike_price):
    value = (stock_price - strike_price) * option_shares
    return max(0, value)


value_option = calculate_options_value(0, 0, 0)
print(f"Valor de las employee Options: ", value_option)
