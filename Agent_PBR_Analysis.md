# Análisis de Heurísticas - Agente Autónomo (PBR)

## 1. Crecimiento Anual de Ingresos (Annual Growth Rate)
El Agente utilizó los reportes de los analistas provistos por Yahoo Finance (`revenueGrowth`) para calcular el primer año de crecimiento. 
- **Crecimiento Base Extraído (Año 1):** 1.00% (Limitado entre 1% y 50% para proteger matemáticamente contra crecimientos exponenciales infinitos).
- **Proyección a 10 Años:** El Agente asume matemáticamente que la ventaja competitiva de la compañía decaerá gradualmente de forma lineal (Year-by-Year) a lo largo de la década. La regresión finaliza convergiendo exactamente con el Risk-Free Rate de la economía en el Año 10.
- **Tasas Aplicadas Calculadas:** Y1: 1.00%, Y2: 1.41%, Y3: 1.82%, Y4: 2.22%, Y5: 2.63%, Y6: 3.04%, Y7: 3.45%, Y8: 3.85%, Y9: 4.26%, Y10: 4.67%

## 2. Margen Operativo (Operating Margin)
Para evitar picos inusuales del último periodo, el Agente buscó el historial en los estados financieros de los **últimos 4 años**.
- Calculó el EBIT sobre Total Revenue de cada periodo histórico disponible y construyó el promedio: **35.14%**. 
- Se asume este Promedio Histórico para modelar la estabilidad del *Core Business* a lo largo de las proyecciones.

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.67%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Oil & Gas Integrated' -> Match más cercano: **'Oil & Gas Integrated'**.
- **Unlevered Beta:** El modelo empleó `-0.218` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de -0.218** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** No encontrados en XLS por fallo estructural, usando proxy ajustado.

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** Brazil.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **21.00%**.
