# Análisis de Heurísticas - Agente Autónomo (JOBY)

## 1. Crecimiento de Ingresos (AGR)
Joby Aviation (JOBY) se clasifica en la etapa de 'Startup / Early High Growth' según la metodología de Aswath Damodaran. La compañía se encuentra en la fase de transición decisiva entre la investigación y desarrollo (I+D) precomercial y el inicio de operaciones comerciales de su flota de aeronaves eléctricas de despegue y aterrizaje vertical (eVTOL). El Mercado Total Direccionable (TAM) para la Movilidad Aérea Urbana (UAM) se proyecta superando $1 billón hacia 2040; Joby posee ventajas significativas para capturar una cuota de mercado sustancial (SOM) respaldada por elevadas barreras de entrada, incluyendo la certificación estricta de la FAA/EASA, patentes propietarias de propulsión eléctrica y alianzas estratégicas estratégicas como Toyota y Uber.

Debido a que la empresa parte de una base de ingresos prácticamente nula, la tasa de crecimiento proyectada exhibe un patrón hiper-acelerado en los primeros años (Front-loaded) a medida que entran en servicio comercial las rutas prioritarias (EE.UU. y Dubai) y contratos gubernamentales/defensa. Conforme el negocio alcance escala industrial y maduración, las tasas de crecimiento anual decrecerán progresivamente, convergiendo de forma asintótica hacia la Tasa Libre de Riesgo de 4.70% en el Año 10.

- **Proyección de Tasas 10A:** `['15000.00%', '500.00%', '200.00%', '100.00%', '50.00%', '30.00%', '20.00%', '12.00%', '7.00%', '4.70%']`

## 2. Margen Operativo
El margen operativo promedio histórico de Joby (-161960.72%) refleja la realidad contable de una empresa pre-comercial con intensos gastos de capital e I+D contrapuestos a ingresos insignificantes. Evaluar a la empresa bajo este promedio histórico distorsionaría el modelo; por ende, se proyecta un proceso de convergencia operado por un elevado apalancamiento operativo a medida que se despliega la flota comercial.

Durante los primeros tres años, los márgenes se mantendrán negativos debido a los costos fijos de fabricación, expansión de infraestructura vertiportuaria y reclutamiento de pilotos. A partir del Año 5, la escala operativa permitirá absorber los costos fijos de R&D y certificaciones, logrando la inflexión hacia terreno positivo. Hacia el Año 10, el Margen Operativo alcanzará un 20.0%, un nivel maduro en línea con el margen diana del sector aeroespacial comercial de alta tecnología y servicios de transporte premium.

- **Proyección de Márgenes 10A:** `['-500.00%', '-150.00%', '-40.00%', '-10.00%', '5.00%', '12.00%', '16.00%', '18.00%', '19.00%', '20.00%']`

## 3. Tasa Libre de Riesgo (Risk-Free Rate) y Equity Risk Premium (ERP)
- **RFR:** El agente rastreó la curva actual de rendimientos macroeconómicos (Símbolo Yahoo: `^TNX`). Valor: **4.70%**.
- **ERP:** Mediante lectura de la web principal de Stern (`home.htm`), extrajo la Prima de Riesgo oficial en vigor: **4.2800%**.
- Estas métricas estandarizan el WACC bajo axiomas actuales de mercado.

## 4. Métricas Sectoriales de Aswath Damodaran
- **Industria base:** 'Airports & Air Services' -> Match más cercano: **'Information Services'**.
- **Unlevered Beta:** El modelo empleó `0.7371769598623608` como normal y eligió re-apalancar usando la **Beta Corregida por Cash de 0.7563563575971642** reconociendo precisamente tus lineamientos de separar los colchones de liquidez de los activos de operación.
- **Sales to Capital Ratio (StCR):** No encontrados en XLS por fallo estructural, usando proxy ajustado.

## 5. Tasa Marginal Estatuaria de Impuestos
- **País Domicilio Legal:** United States.
- Basado en el marco geográfico, el Agente descarta la engañosa tasa de "interés efectivo" a la que las empresas suelen camuflar temporalmente sus ganancias, y prefiere imponer una tasa legal marginal directa para proyección a largo plazo: **25.00%**.
