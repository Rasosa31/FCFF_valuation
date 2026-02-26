import numpy as np
import matplotlib.pyplot as plt

# Parámetros base
ingresos_iniciales = 100000000  # $100M
tasa_impuestos = 0.25
capex_porcentaje = 0.10
cambio_capital_trabajo_porcentaje = 0.05
anios = 5
deuda_neta = 500000
acciones = 1000000
n_simulaciones = 10000

# Distribuciones para variables inciertas
np.random.seed(42)
margen_ebit = np.random.normal(0.15, 0.02, n_simulaciones)  # Media 15%, desv. 2%
wacc = np.random.normal(0.08, 0.01, n_simulaciones)  # Media 8%, desv. 1%
crecimiento_perpetuo = np.random.normal(
    0.02, 0.005, n_simulaciones
)  # Media 2%, desv. 0.5%


# Función para calcular FCFF
def calcular_fcff(
    ingresos,
    margen_ebit,
    tasa_impuestos,
    capex_porcentaje,
    cambio_capital_trabajo_porcentaje,
):
    ebit = ingresos * margen_ebit
    fcff = (
        ebit * (1 - tasa_impuestos)
        - (ingresos * capex_porcentaje)
        - (ingresos * cambio_capital_trabajo_porcentaje)
    )
    return fcff


# Función para calcular el valor de la empresa
def valor_empresa(fcff_proyectados, wacc, crecimiento_perpetuo, anios):
    # Debugging: Print problematic WACC vs. g scenarios
    if wacc <= crecimiento_perpetuo:
        # print(f"DEBUG: WACC ({wacc:.4f}) <= Crecimiento Perpetuo ({crecimiento_perpetuo:.4f})")
        return None
    if (
        not fcff_proyectados
    ):  # This case should be handled by len(fcff_proyectados) == anios check
        # print("DEBUG: fcff_proyectados is empty.")
        return None

    valor = 0
    for t in range(1, anios + 1):
        valor += fcff_proyectados[t - 1] / (1 + wacc) ** t

    # Calculate Terminal Value correctly
    valor_terminal = (
        fcff_proyectados[-1]
        * (1 + crecimiento_perpetuo)
        / (wacc - crecimiento_perpetuo)
    )
    valor += valor_terminal / (1 + wacc) ** anios
    return valor


# Simulación de Montecarlo
valores_por_accion = []
discarded_due_to_wacc_g = 0
discarded_due_to_negative_equity = 0
discarded_due_to_negative_price = 0
discarded_due_to_fcff_length = 0

for i in range(n_simulaciones):
    # Proyección de ingresos (crecimiento aleatorio entre 3% y 7%)
    tasa_crecimiento_ingresos = np.random.uniform(0.03, 0.07)
    ingresos = [ingresos_iniciales]
    for _ in range(anios):
        ingresos.append(ingresos[-1] * (1 + tasa_crecimiento_ingresos))

    # Calcular FCFF para cada año de proyección (desde el año 1)
    fcff_proyectados = []
    for ing_anual in ingresos[1:]:
        fcff = calcular_fcff(
            ing_anual,
            margen_ebit[i],
            tasa_impuestos,
            capex_porcentaje,
            cambio_capital_trabajo_porcentaje,
        )
        fcff_proyectados.append(fcff)

    # Validate that fcff_proyectados has the correct number of elements
    if len(fcff_proyectados) != anios:
        discarded_due_to_fcff_length += 1
        continue  # Skip this simulation if FCFF length is wrong

    # Calcular valor de la empresa
    valor = valor_empresa(fcff_proyectados, wacc[i], crecimiento_perpetuo[i], anios)

    if valor is None:
        discarded_due_to_wacc_g += 1  # This counts cases where WACC <= g
        continue

    # Valor de la equidad y precio por acción
    if valor <= deuda_neta:  # If enterprise value is less than or equal to net debt
        discarded_due_to_negative_equity += 1
        continue

    valor_equidad = valor - deuda_neta
    precio_accion = valor_equidad / acciones

    if precio_accion <= 0:  # If price per share is non-positive
        discarded_due_to_negative_price += 1
        continue

    valores_por_accion.append(precio_accion)

print(f"\n--- Simulation Summary ---")
print(f"Total simulations: {n_simulaciones}")
print(f"Simulations included in results: {len(valores_por_accion)}")
print(f"Discarded (FCFF length mismatch): {discarded_due_to_fcff_length}")
print(f"Discarded (WACC <= Crecimiento Perpetuo): {discarded_due_to_wacc_g}")
print(f"Discarded (Valor Empresa <= Deuda Neta): {discarded_due_to_negative_equity}")
print(f"Discarded (Precio por Acción <= 0): {discarded_due_to_negative_price}")

# Análisis de resultados
if valores_por_accion:
    media = np.mean(valores_por_accion)
    mediana = np.median(valores_por_accion)
    percentil_5 = np.percentile(valores_por_accion, 5)
    percentil_95 = np.percentile(valores_por_accion, 95)

    print(f"\nValor justo por acción (media): ${media:.2f}")
    print(f"Valor justo por acción (mediana): ${mediana:.2f}")
    print(
        f"Intervalo de confianza 90% (5º-95º percentil): ${percentil_5:.2f} - ${percentil_95:.2f}"
    )

    # Visualización
    plt.hist(valores_por_accion, bins=50, edgecolor="black")
    plt.axvline(media, color="red", linestyle="--", label=f"Media: ${media:.2f}")
    plt.axvline(
        percentil_5,
        color="green",
        linestyle="--",
        label=f"5º Percentil: ${percentil_5:.2f}",
    )
    plt.axvline(
        percentil_95,
        color="green",
        linestyle="--",
        label=f"95º Percentil: ${percentil_95:.2f}",
    )
    plt.xlabel("Valor por acción ($)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del valor justo por acción (Simulación Montecarlo)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(
        "No se generaron valores válidos. Revisa los parámetros de entrada o las condiciones de filtrado."
    )
