# Análisis de Heurísticas - Agente Autónomo (INTC)

## 1. Crecimiento Anual de Ingresos (Annual Growth Rate)
El Agente utilizó los reportes de los analistas provistos por Yahoo Finance (`revenueGrowth`) para calcular el primer año de crecimiento. 
- **Crecimiento Base Extraído (Año 1):** 7.20% (Limitado entre 1% y 50% para proteger matemáticamente contra crecimientos exponenciales infinitos).
- **Proyección a 10 Años:** El Agente asume matemáticamente que la ventaja competitiva de la compañía decaerá gradualmente de forma lineal (Year-by-Year) a lo largo de la década. La regresión finaliza convergiendo exactamente con el Risk-Free Rate de la economía en el Año 10.
- **Tasas Aplicadas Calculadas:** Y1: 7.20%, Y2: 6.91%, Y3: 6.63%, Y4: 6.34%, Y5: 6.06%, Y6: 5.77%, Y7: 5.49%, Y8: 5.20%, Y9: 4.91%, Y10: 4.63%

## 2. Margen Operativo (Operating Margin)
Para evitar picos inusuales del último periodo, el Agente buscó el historial en los estados financieros de los **últimos 4 años**.
- Calculó el EBIT sobre Total Revenue de cada periodo histórico disponible y construyó el promedio: **1.00%**. 
- Se asume este Promedio Histórico para modelar la estabilidad del *Core Business* a lo largo de las proyecciones.

## 3. Tasa Libre de Riesgo (Risk-Free Rate / RFR)
- El agente rastreó en vivo la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). 
- **Valor actual capturado de bonos del Tesoro EE.UU a 10 años:** **4.63%**.
- Esta métrica no sólo sustenta el WACC, sino que también ejerce como ancla inamovible para la Tasa Perpetua de Crecimiento del Año Terminal (Terminal Growth).

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base inferida:** 'Semiconductors'
- Al conectarse con NYU Stern, el Agente aplicó *algoritmos de coincidencia aproximada (Fuzzy Match)* y alineó el perfil de la compañía con la industria de Damodaran más cercana: **'Semiconductors'**. Luego extrajo:
  - **Unlevered Beta:** 2.187 (Sustituyendo silenciosamente el factor altamente volátil apalancado predeterminado que arroja la bolsa).
  - **Sales to Capital Ratio (StCR):** Datos no encontrados en XLS por fallo de conexión, usando proxy predeterminado ajustado.

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
