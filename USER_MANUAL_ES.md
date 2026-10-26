# Manual de Usuario – Aplicación de Valoración FCFF

## 1. Introducción
Esta aplicación permite estimar el valor de una empresa mediante el método **Flujo de Caja Libre para la Firma (FCFF)**. Está diseñada para ser utilizada por analistas financieros, estudiantes y cualquier persona interesada en valorar compañías de forma estructurada y reproducible.

## 2. Requisitos del Sistema
- **Python 3.9+**
- **Streamlit** (interfaz web)
- Dependencias especificadas en `requirements.txt` (p.ej., `pandas`, `numpy`, `openpyxl`, `plotly`, etc.)
- Navegador web moderno (Chrome, Firefox, Safari, Edge)

## 3. Instalación
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/fcff-valuation-app.git
   cd fcff-valuation-app
   ```
2. **Crear un entorno virtual** (opcional pero recomendado)
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # .\venv\Scripts\activate  # Windows
   ```
3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```
   La app se abrirá automáticamente en el navegador en la dirección `http://localhost:8501`.

## 4. Descripción de la Interfaz
La aplicación se compone de una barra lateral (sidebar) con diversas secciones y una zona principal con pestañas.

### 4.1 Barra lateral
| Sección | Parámetro | Descripción | Valor por defecto |
|---------|-----------|-------------|-------------------|
| **Datos Base** | `Revenue` | Ingresos del año 0 (en la moneda elegida). | 100,000 |
| | `EBIT` | Earnings Before Interest and Taxes del año 0. | 15,000 |
| | `Tax Rate (t)` | Tasa impositiva esperada (decimal). | 0.21 |
| | `Invested Capital` | Capital invertido inicial. | 90,000 |
| **Año Terminal** | `Terminal Year` | Año en el que se calcula el valor terminal. | 10 |
| | `Terminal Growth Rate (g)` | Tasa de crecimiento perpetuo del ingreso en el año terminal. **Valor por defecto: 0.03 (3 %)**. | 0.03 |
| **Costo de Capital** | `Risk‑Free Rate (RFR)` | Tasa libre de riesgo. | 0.025 |
| | `Equity Risk Premium (ERP)` | Prima de riesgo de capital. | 0.055 |
| | `Beta` | Beta del mercado (si está fuera del rango 0.5‑2.0 se sustituye por el beta sectorial). | – |
| **Mercado & Acciones** | `Shares Outstanding` | Número total de acciones. | 1,000,000 |
| | `Current Share Price` | Precio actual de la acción. | 50 |
| **Gastos de I+D** | `R&D Expense` | Gasto anual de investigación y desarrollo. | 0 |
| **Deuda & Opciones** | `Debt` | Deuda neta. | 0 |
| | `Option Value` | Valor de opciones sobre la empresa. | 0 |

> **Nota:** Todos los campos aceptan valores decimales o porcentuales según corresponda. La nueva variable **`g`** representa la tasa de crecimiento terminal y se utiliza en varios cálculos (valor terminal, reinversión, descuento).

### 4.2 Pestañas principales
- **Tab 1 – Resumen de Valuación**: muestra los principales indicadores (Valor de la Firma, Valor por Acción, Valor Terminal, WACC, ROIC, etc.) y gráficos de proyección.
- **Tab 2 – Monte‑Carlo**: simulación de sensibilidad basada en distribuciones normales para RFR, ERP y otros parámetros.
- **Tab 3 – Tabla de Resultados**: DataFrame con los valores anuales de ingresos, EBIT, impuestos, flujo de caja, capital invertido, ROIC, reinversión, etc.
- **Tab 4 – Exportar**: permite descargar los resultados en **Excel** o **JSON** para su posterior análisis.

## 5. Procedimiento de Uso
1. **Cargar datos** (opcional): use el botón *Load CSV* para importar un archivo con los datos históricos de la empresa.
2. **Ajustar los parámetros** en la barra lateral según la información disponible.
3. Pulsar el botón **“Run Valuation”**.
4. Revisar los resultados en la pestaña *Resumen de Valuación* y las gráficas.
5. Si se desea explorar la variabilidad, ir a la pestaña *Monte‑Carlo* y observar los histogramas y estadísticas.
6. Exportar los resultados mediante los botones **Download Excel** o **Save JSON** para compartir o guardar la valoración.

## 6. Interpretación de Resultados
- **Valor de la Firma (Enterprise Value)**: valor total de la empresa antes de considerar deuda y efectivo.
- **Valor por Acción (Equity Value / Shares)**: precio implícito de cada acción.
- **Valor Terminal**: valor presente de los flujos de caja después del año terminal, calculado como `Terminal CF / (WACC - g)`.
- **WACC**: Costo medio ponderado de capital, combinación de costo de deuda y costo de capital propio.
- **ROIC**: **Return on Invested Capital** calculado ahora como `EBIT(1‑t) / Inv_Capital` **del mismo año** (sin retraso). Esta columna se muestra en la tabla de resultados.
- **Reinversión (Reinvestment)**: proporción del crecimiento terminal que se destina a nuevas inversiones, calculada como `g / ROIC_terminal`.
- **Beta del Modelo (Metodología Damodaran)**: El motor utiliza la **Unlevered Beta corregida por caja** del sector de la compañía (obtenida de la base de datos de Aswath Damodaran en NYU Stern, archivo `betas.xls`, hoja *Industry Averages*, columna P: *Average (2022-2026)*) y la apalanca con la estructura de capital de la compañía a través de la fórmula de Hamada:
  $$\beta_L = \beta_{U,\text{corrected}} \times \left[1 + (1 - T) \times \frac{D}{MCap}\right]$$
  La beta directa de Yahoo Finance se preserva como dato informativo de referencia en el informe.

## 7. Consejos y Buenas Prácticas
- Ajuste la **tasa de crecimiento `g`** únicamente cuando tenga una estimación fundamentada; el valor por defecto (3 %) es una referencia genérica.
- Verifique que la **tasa libre de riesgo (RFR)** corresponda al horizonte temporal de la valoración.
- La **Beta apalancada** refleja tanto el riesgo operativo de la industria como el riesgo financiero derivado de la deuda de la empresa ($D / MCap$). Si la empresa no cotiza o no tiene deuda, $\beta_L = \beta_{U,\text{corrected}}$.
- Utilice el **exportado JSON** para guardar configuraciones personalizadas y cargarlas posteriormente.
- Revise la sección *Monte‑Carlo* para entender la sensibilidad de los resultados ante cambios en RFR, ERP y `g`.

## 8. Solución de Problemas
| Síntoma | Causa posible | Solución |
|---------|----------------|----------|
| **Error “module not found”** al iniciar la app | Dependencias faltantes | Ejecutar `pip install -r requirements.txt` dentro del entorno virtual. |
| **Valores NaN** en la tabla | Entrada numérica vacía o no numérica | Asegúrese de que todos los campos tengan valores válidos (usar `0` si no aplica). |
| **La aplicación no se abre** después de `streamlit run app.py` | Puerto 8501 ocupado | Cambiar el puerto: `streamlit run app.py --server.port 8502` |
| **Beta fuera de rango y no se reemplaza** | Archivo `betas.xls` no está en la ruta esperada | Colocar `betas.xls` en la carpeta `data/` del proyecto o actualizar la ruta en `valuation_engine.py`. |
| **Errores al exportar Excel** | Falta de `openpyxl` | Instalar con `pip install openpyxl`. |

## 9. Exportación / Importación
- **Exportar a Excel**: botón *Download Excel* genera `valuation_results.xlsx` con todas las tablas y gráficos.
- **Guardar configuración en JSON**: botón *Save JSON* crea `valuation_config.json`. Para cargar una configuración previa, use *Load JSON*.

## 10. Licencia y Contacto
Este proyecto está bajo licencia **MIT**. Para reportar errores, solicitar mejoras o colaborar, abra un *issue* en el repositorio GitHub o contacte al autor vía correo electrónico `contacto@ejemplo.com`.

---
*Este manual ha sido generado automáticamente por Antigravity, la asistente de programación de Google DeepMind.*
