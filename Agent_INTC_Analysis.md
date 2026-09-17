# Análisis de Heurísticas - Agente Autónomo (INTC)

## 1. Crecimiento de Ingresos (AGR)
La trayectoria de ingresos proyectada para Intel inicia con un marcado rebote cíclico del 25.40% en el Año 1, alineado con el consenso de analistas, impulsado por la recuperación del mercado de PCs (ciclo de reemplazo con AI PCs), la reactivación del gasto en servidores y una baja base de comparación tras el reciente valle operativo.

Durante los Años 2 a 5, el crecimiento se modera de forma no lineal pero se mantiene robusto (17.5% bajando a 7.0%), sustentado en la rampa de producción del nodo Intel 18A y los primeros ingresos significativos de Intel Foundry Services (IFS) al ganar clientes externos. No obstante, la férrea competencia de TSMC, AMD y NVIDIA evita una aceleración prolongada.

En los Años 6 a 10, la tasa de expansión se desacelera de manera gradual hasta converger racionalmente con la tasa libre de riesgo (4.67%). En esta fase madura, Intel estabiliza su cuota de mercado global en un modelo híbrido de diseño de chips y servicios de fundición a escala global.

- **Proyección de Tasas 10A:** `['25.40%', '17.50%', '12.00%', '9.00%', '7.00%', '5.80%', '5.20%', '4.80%', '4.67%', '4.67%']`

## 2. Margen Operativo
El margen operativo de Intel evoluciona desde una cifra histórica reciente deprimida (-1.29%), afectada por severos cargos de reestructuración, altos costos de desarrollo de nodos acelerados y pérdidas de escala inicial en la división de fundición. Proyectar un margen estático a perpetuidad distorsionaría el valor real de la empresa.

En la primera fase (Años 1-4), el margen operativo experimenta un cambio de rumbo (turnaround) dinámico, pasando de 3.5% a 16.0%. Esta recuperación sustancial está impulsada por el programa agresivo de reducción de costos operativos ($10B+), la adopción de la litografía EUV que elimina capas de trabajo costosas, y la implementación del modelo de fundición interna que impone disciplina de costos y eficiencias comerciales entre unidades de negocio.

A partir del Año 5 y hasta el Año 8, el margen alcanza su meseta de alta rentabilidad (entre 18.5% y 21.0%), beneficiándose de economías de escala plenas en el nodo 18A/14A y del apalancamiento operativo del negocio de foundry. En los Años 9 y 10, el margen se ajusta levemente al 20.0% para reflejar la presión competitiva continua de precios y los requerimientos recurrentes de reinversión en capital (CapEx/R&D) necesarios para mantener el liderazgo tecnológico.

- **Proyección de Márgenes 10A:** `['3.50%', '8.00%', '12.50%', '16.00%', '18.50%', '20.00%', '21.00%', '21.00%', '20.50%', '20.00%']`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.67%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Semiconductors' -> Match más cercano: **'Semiconductor'**.
- **Unlevered Beta:** El modelo empleó `1.48925822268736` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de 1.5046492754744247** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** 1.206668138058751

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
