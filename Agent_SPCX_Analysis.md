# Análisis de Heurísticas - Agente Autónomo (SPCX)

## 1. Crecimiento de Ingresos (AGR)
SPCX se clasifica bajo la metodología de Aswath Damodaran en la etapa de 'High Growth' (Crecimiento Acelerado), transitando desde una fase de comercialización inicial hacia la escala industrial dentro del sector Aerospace/Defense. Con un crecimiento de consenso para el Año 1 de 91.90%, la empresa demuestra una fuerte adopción de mercado respaldada por contratos gubernamentales y comerciales en expansión. Sin embargo, dada la maduración gradual de la base de ingresos y los largos ciclos de procuración del sector, se proyecta un patrón de desaceleración tipo 'Front-loaded'.

El mercado direccionable (TAM) en el sector de defensa y aeroespacial cuenta con barreras de entrada extremadamente elevadas (certificaciones de seguridad, intensivas inversiones en I+D y relaciones consolidadas con clientes institucionales). A medida que SPCX capture una mayor porción del mercado obtenible (SOM), el crecimiento anual decrecerá de manera asintótica hacia la Tasa Libre de Riesgo (4.67%) para el Año 10, reflejando el comportamiento típico de una empresa aeroespacial madura que crece a la par de la economía global.

- **Proyección de Tasas 10A:** `['91.90%', '52.00%', '30.00%', '18.00%', '12.00%', '8.50%', '6.50%', '5.20%', '4.80%', '4.67%']`

## 2. Margen Operativo
La trayectoria del Margen Operativo (OPM) para SPCX parte de un promedio histórico de -0.29%, característico de empresas en fase de aceleración con altos costos fijos en I+D y escalamiento de infraestructura de fabricación. A medida que la compañía aumente su volumen de producción y acelere las entregas, la absorción de costos fijos generará una importante apalancamiento operativo ('Operating Leverage').

Se proyecta que el margen operativo evolucione desde terreno neutral/ligeramente positivo en el Año 1 (2.00%) hasta alcanzar la media del sector de aeroespacio y defensa maduro (12.00%) hacia el Año 10. La protección de patentes, los altos costos de cambio para los clientes y las economías de escala sostendrán este margen frente a presiones competitivas a largo plazo.

- **Proyección de Márgenes 10A:** `['2.00%', '4.00%', '6.00%', '8.00%', '9.50%', '10.50%', '11.20%', '11.70%', '12.00%', '12.00%']`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.67%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Aerospace & Defense' -> Match más cercano: **'Aerospace/Defense'**.
- **Unlevered Beta:** El modelo empleó `0.846668742124111` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de 0.8693772823914925** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** No encontrados en XLS por fallo estructural, usando proxy ajustado.

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
