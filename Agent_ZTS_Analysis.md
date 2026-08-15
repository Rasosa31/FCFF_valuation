# Análisis de Heurísticas - Agente Autónomo (ZTS)

## 1. Crecimiento de Ingresos (AGR)
Zoetis Inc. (ZTS), evaluada dentro del sector Retail (Special Lines), se clasifica formalmente bajo la etapa de 'Mature Growth' derivado de su posición de liderazgo consolidada pero ajustada a la desaceleración del entorno macroeconómico actual. Con una contracción prospectiva consensuada para el Año 1 del -0.20%, la empresa enfrenta vientos en contra a corto plazo caracterizados por la normalización del gasto del consumidor y la reestructuración de inventarios en líneas especializadas.

A nivel de Mercado Total Direccionable (TAM) y SOM, ZTS retiene ventajas competitivas sostenibles gracias a patentes clave, lealtad de marca y elevadas barreras de entrada técnicas e higiénicas. Sin embargo, la madurez del sector limita la capacidad de captura de nueva cuota a ritmos acelerados, haciendo que el crecimiento a largo plazo dependa principalmente del poder de fijación de precios y de innovaciones incrementales en líneas especializadas.

La trayectoria de ingresos proyectada refleja una recuperación gradual tras la leve caída del Año 1 (-0.20%). A partir del Año 2, los ingresos rebotan hacia niveles moderados del 3.5% al 5.0% conforme se estabiliza la demanda, para posteriormente desacelerarse de manera asintótica hacia la tasa libre de riesgo del 4.64% hacia el Año 10, garantizando la consistencia teórica del modelo de crecimiento terminal.

- **Proyección de Tasas 10A:** `['-0.20%', '3.50%', '4.50%', '5.00%', '4.80%', '4.70%', '4.65%', '4.64%', '4.64%', '4.64%']`

## 2. Margen Operativo
El margen operativo histórico de ZTS se ubica en un sobresaliente 36.70%, muy por encima del promedio del comercio minorista tradicional debido a la alta especialización y poder de fijación de precios de sus líneas de productos. Para el Año 1, se proyecta una leve compresión del margen al 36.00% como consecuencia del deleverage operativo derivado de la reducción esperada en los ingresos (-0.20%).

A medida que el crecimiento del volumen se restablece entre los años 2 y 5, las economías de escala y la optimización de la cadena de suministro permitirán una expansión progresiva del margen operativo, alcanzando un pico proyectado del 37.00% en el mediano plazo. Esta mejora captura la capacidad de la empresa para diluir costos fijos sobre una base de ingresos revitalizada.

Hacia la segunda mitad de la década (Años 6 al 10), la presión competitiva en el segmento Retail especializado y la maduración de líneas clave actuarán como techo estructural. Los márgenes operativos convergerán suavemente de vuelta hacia su promedio histórico de 36.70%, asegurando un estado estacionario altamente rentable y sostenible para el flujo de caja libre a la firma.

- **Proyección de Márgenes 10A:** `['36.00%', '36.30%', '36.50%', '36.70%', '36.80%', '37.00%', '37.00%', '36.90%', '36.80%', '36.70%']`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.64%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Drug Manufacturers - Specialty & Generic' -> Match más cercano: **'Retail (Special Lines)'**.
- **Unlevered Beta:** El modelo empleó `0.9487830646597303` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de 1.0018775790318082** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** 2.98844717708956

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
