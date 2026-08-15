# Análisis de Heurísticas - Agente Autónomo (AAPL)

## 1. Crecimiento de Ingresos (AGR)
Apple Inc. (AAPL) se clasifica en la etapa de 'Crecimiento Maduro' (Mature Growth) según la metodología de ciclo de vida de Damodaran. La compañía cuenta con una base instalada activa superior a los 2.200 millones de dispositivos que actúa como un foso económico defensivo invulnerable, impulsado por elevados costes de cambio y el bloqueo de ecosistema (iOS). Si bien el mercado de hardware tradicional se encuentra saturado, la captura de valor se traslada hacia el incremento del ingreso promedio por usuario (ARPU) y la monetización del segmento de servicios.

- **Proyección de Tasas 10A:** `['16.40%', '13.50%', '11.00%', '9.00%', '7.50%', '6.50%', '5.80%', '5.20%', '4.80%', '4.66%']`

## 2. Margen Operativo
El margen operativo (OPM) se proyecta dinámicamente evaluando dos fuerzas contrapuestas. Por un lado, la expansión del segmento de Servicios (que opera con márgenes brutos superiores al 70%) y las eficiencias de escala derivadas de los chips propios (Apple Silicon) impulsarán los márgenes al alza durante la primera mitad del periodo prospectivo, superando ligeramente el promedio histórico del 30.90%. Por otro lado, presiones regulatorias globales (como el escrutinio antimonopolio sobre la App Store) y un mayor gasto estructural en I+D para sostener la infraestructura de Inteligencia Artificial tenderán a erosionar parcialmente esa expansión en la segunda mitad del horizonte. El recorrido final estabiliza los márgenes en torno a su media histórica empírica del 30.90% hacia el final de la década.

- **Proyección de Márgenes 10A:** `['31.20%', '31.80%', '32.30%', '32.50%', '32.40%', '32.10%', '31.80%', '31.50%', '31.20%', '30.90%']`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.66%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Consumer Electronics' -> Match más cercano: **'Computer Services'**.
- **Unlevered Beta:** El modelo empleó `0.9155102559970569` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de 0.9617009989891427** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** No encontrados en XLS por fallo estructural, usando proxy ajustado.

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
