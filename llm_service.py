import json
import os
import re
import time
from datetime import date
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Modelos recomendados de Gemini en orden de prioridad
GEMINI_MODELS = ['gemini-3.8-flash', 'gemini-3.8-pro']


def get_llm_projections(
    ticker,
    industry,
    current_margins,
    rfr,
    base_revenue_growth,
    # --- Parámetros enriquecidos del negocio ---
    long_name: str = "",
    business_summary: str = "",
    sector: str = "",
    country: str = "",
    revenue_history: list = None,      # lista de tuplas (año, ingresos_M) más reciente primero
    margin_history: list = None,       # lista de tuplas (año, margen_op) más reciente primero
    beta: float = None,
    sector_beta: float = None,         # Beta sectorial Damodaran si beta es outlier
    interest_expense: float = None,    # en $M
    interest_expense_source: str = "Yahoo Finance",
    valuation_date: str = None,
):
    """
    Invoca la API de Gemini para generar un análisis fundamental profundo,
    incluyendo el Perfil Completo de la Compañía (narrativa escrita) y las proyecciones
    no lineales a 10 años para Crecimiento de Ingresos (AGR) y Margen Operacional (OPM).
    """
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key or api_key == "PEGA_AQUÍ_TU_API_KEY":
        print("❌ API Key no encontrada en .env ni st.secrets.")
        return None

    print("🧠 Invocando al Modelo Gemini Advanced (aistudio.google.com)...")

    today = valuation_date or str(date.today())

    # ── Formato de beta ───────────────────────────────────────────────────────
    beta_context = ""
    if beta is not None:
        if beta < 0.5 or beta > 2.0:
            beta_context = (
                f"\n⚠️ NOTA BETA: El beta de mercado reportado ({beta:.2f}) se considera un outlier "
                f"(fuera del rango 0.5 - 2.0). El Agente sustituirá este valor por el beta sectorial "
                f"({sector_beta:.4f} asignar beta del sector usando la fuente de Mr Damodaran archivo betas.xls columna C). "
                f"Explica brevemente esto en el campo 'beta_note'."
            )
        else:
            beta_context = f"\n- Beta Levered de la compañía: {beta:.2f} (dentro del rango aceptable)."

    # ── Formato de series históricas ──────────────────────────────────────────
    rev_hist_str = "No disponible"
    if revenue_history:
        rev_hist_str = "\n".join(
            f"   - {yr}: ${rev:,.1f}M" for yr, rev in revenue_history
        )

    margin_hist_str = "No disponible"
    if margin_history:
        margin_hist_str = "\n".join(
            f"   - {yr}: {m:.2%}" for yr, m in margin_history
        )

    # ── Resumen del negocio ───────────────────────────────────────────────────
    biz_summary_str = f"\nResumen Operativo / Descripción del Negocio:\n{business_summary[:1200]}..." if business_summary else ""

    # ── Nota de gastos por intereses ──────────────────────────────────────────
    int_exp_note = ""
    if interest_expense is not None and interest_expense == 0.0:
        int_exp_note = (
            "\n⚠️ NOTA GASTOS POR INTERESES: No se identificaron gastos por intereses separados en los datos base. "
            "En 'interest_expense_note' explica que si la empresa tiene deuda financiera se debe consultar el 10-K "
            "o estimar con cupones de bonos emitidos."
        )

    full_prompt = f"""
Eres un Director Senior de Valoración Fundamental y Corporate Finance de élite, especializado en el método FCFF (Free Cash Flow to Firm) de Aswath Damodaran.
Tu misión es doble y de la más alta exigencia profesional:
1. Redactar una **NARRATIVA ESCRITA COMPLETA Y DETALLADA** de la compañía que proporcione todo el contexto estratégico y cualitativo que fundamenta la valoración.
2. Construir las proyecciones dinámicas y no lineales a 10 años para el **Crecimiento Anual de Ingresos (AGR)** y el **Margen Operativo (OPM)**, asegurando que los números reflejen fielmente la historia estratégica de la empresa.

══════════════════════════════════════════════════════════════════════
DATOS DE ENTRADA Y CONTEXTO
══════════════════════════════════════════════════════════════════════
- Compañía: {long_name or ticker} (Ticker: {ticker})
- Sector / Industria: {sector} | {industry}
- País / Domicilio Legal: {country or 'United States'}
- Fecha de Valoración: {today}
- Margen Operativo Histórico Reciente: {current_margins:.2%}
- Tasa Libre de Riesgo (RFR): {rfr:.2%}
- Crecimiento Estimado de Consenso Analistas (Año 1): {base_revenue_growth:.2%}
{biz_summary_str}

Historial de Ingresos Anuales (más reciente primero):
{rev_hist_str}

Historial de Margen Operativo (más reciente primero):
{margin_hist_str}
{beta_context}
{int_exp_note}

══════════════════════════════════════════════════════════════════════
DIRECTRICES ESPECÍFICAS PARA EL CONTENIDO
══════════════════════════════════════════════════════════════════════

### 1. NARRATIVA ESCRITA: PERFIL DE LA COMPAÑÍA (campo "company_profile")
Debes redactar un texto exhaustivo en Markdown que incluya OBLIGATORIAMENTE cada una de estas 6 sub-secciones con sus títulos respectivos:

- **Descripción de la Compañía**: Explica en detalle qué hace la empresa, sus líneas de negocio principales, fuentes de ingresos, modelo operativo, presencia geográfica, y su posicionamiento en el mercado global. Incluye la fecha de elaboración ({today}).
- **Ciclo de Vida de la Empresa**: Clasifica de forma fundamentada a la empresa en una de las 6 etapas del ciclo de vida corporativo: *Startup*, *Young Growth*, *High Growth*, *Mature Growth*, *Mature Stable*, o *Decline*. Justifica con argumentos de penetración de mercado, reinversión y madurez de la industria.
- **Ambiente Competitivo y Retos**: Analiza el entorno competitivo (rivalidad, amenazas de sustitutos, barreras de entrada, intensidad de I+D), riesgos regulatorios, geopolíticos (ej. seguridad nacional, subsidios como el CHIPS Act), y los principales retos estratégicos a superar.
- **Ventaja Competitiva Sostenible (Moat)**: Identifica si la empresa cuenta con un Moat económico (efecto de red, patentes/propiedad intelectual, costos de cambio, ventajas de escala, marcas). Describe sus componentes y **valora cuantitativamente su aporte** (ej. "Aporta entre 150 y 300 puntos básicos de protección al margen operativo frente a competidores directos"). Si no tiene moat, explícalo.
- **Fuentes de Extracción de Datos**: Detalla explícitamente las fuentes consultadas y utilizadas para el análisis: Yahoo Finance (estados financieros consolidados y precios de mercado), SEC EDGAR (reportes 10-K y 10-Q), NYU Stern / Damodaran (parámetros sectoriales de Sales to Capital, ERP y Betas), consensos de analistas de Wall Street y estimaciones directas de la gerencia (earnings calls).
- **Conexión Narrativa con los Números**: Conecta explícitamente los eventos cualitativos de la empresa con las cifras financieras observadas (ej. caídas de margen por write-downs de fábricas o inversiones masivas de capital vs. recuperación de márgenes por nuevos productos, cambio de liderazgo, inyección de capitales o eficiencias operativas).

### 2. PROYECCIÓN DINÁMICA DE INGRESOS - AGR (campo "agr_list" y "revenue_narrative")
- Genera una lista de exactamente 10 tasas decimales para los años 1 al 10 (`agr_list`).
- **REGLA CRÍTICA DE CONVERGENCIA**: **NO FUERCES que en el año 10 el crecimiento de ingresos confluya rígidamente con la tasa libre de riesgo ({rfr:.2%})**. Si la compañía (incluso si es madura o en turnaround) cuenta con catalizadores estructurales (boom de IA, subsidios gubernamentales, nuevos mercados, capacidad fabril adicional, liderazgo renovado), es plenamente legítimo y esperado que en el Año 10 siga creciendo significativamente por encima de la tasa terminal (por ejemplo entre 6% y 12%+). No aceleres artificialmente la desaceleración.
- Prohíbe cualquier decrecimiento lineal automático o plano. Justifica en `revenue_narrative` las distintas fases de aceleración, estabilización o maduración.

### 3. PROYECCIÓN DINÁMICA DE MARGEN OPERACIONAL - OPM (campo "opm_list" y "margin_narrative")
- Genera una lista de exactamente 10 márgenes operativos decimales para los años 1 al 10 (`opm_list`).
- **PROHIBICIÓN TOTAL DEL PROMEDIO HISTÓRICO ESTÁTICO**: Bajo ninguna circunstancia uses un margen plano o un promedio histórico estático.
- **CASOS DE TURNAROUND / MÁRGENES DEPRIMIDOS**: Si la empresa ha experimentado márgenes operativos negativos o comprimidos en los últimos años debido a reestructuraciones, depreciaciones agresivas, write-downs o transición de modelos de negocio (como en el caso de Intel), DEBES modelar una **CURVA DE RECUPERACIÓN PROGRESIVA Y EXPANSIÓN DINÁMICA DE MÁRGENES** hacia los niveles sostenibles y competitivos de su industria (15% a 25%+ según corresponda).
- Incorpora en la trayectoria año a año los efectos de:
  1. Nuevos productos y vientos de cola sectoriales (ej. IA, demanda de servidores y semiconductores avanzados).
  2. Subsidios gubernamentales e inyecciones de capital (ej. CHIPS Act).
  3. Reestructuración de costos, venta de divisiones no rentables y eficiencias de escala.
  4. Maduración de nuevas fábricas e inversiones en capacidad instalada.
- Justifica ampliamente esta recuperación en `margin_narrative`.

══════════════════════════════════════════════════════════════════════
ESTRUCTURA DE RESPUESTA — FORMATO JSON OBLIGATORIO
══════════════════════════════════════════════════════════════════════
Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
{{
    "company_profile": "# Perfil Estratégico de la Compañía: {long_name or ticker}\\n\\n**Fecha de elaboración:** {today}\\n\\n### 1. Descripción de la Compañía\\n[Texto amplio y descriptivo]\\n\\n### 2. Ciclo de Vida de la Empresa\\n[Texto identificando y justificando el ciclo de vida]\\n\\n### 3. Ambiente Competitivo y Retos\\n[Texto analizando el entorno competitivo, barreras y retos]\\n\\n### 4. Ventaja Competitiva Sostenible (Moat)\\n[Texto describiendo el Moat y cuantificando su aporte al margen/crecimiento]\\n\\n### 5. Fuentes de Extracción de Datos\\n- Yahoo Finance: Estados financieros consolidados (Income Statement TTM, Balance Sheet trimestral)\\n- SEC EDGAR: Informes periódicos 10-K y notas a los estados financieros\\n- NYU Stern (Aswath Damodaran): Base de datos sectorial 2026 (Sales-to-Capital, ERP, Betas)\\n- Informes de analistas y conferencias de resultados (Earnings Calls)\\n\\n### 6. Conexión Narrativa con los Números\\n[Texto conectando la historia con los datos históricos y las proyecciones futuras]",
    "revenue_narrative": "Explicación detallada de 3-4 párrafos que justifica la trayectoria año a año de los ingresos proyectados, destacando por qué no se acelera la confluencia terminal.",
    "agr_list": [0.12, 0.14, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.08, 0.075],
    "margin_narrative": "Explicación detallada de 3-4 párrafos que fundamenta la evolución dinámica y recuperación del margen operativo año a año (turnaround, eficiencias, subsidios, pricing power).",
    "opm_list": [0.03, 0.07, 0.12, 0.16, 0.19, 0.21, 0.23, 0.24, 0.25, 0.25],
    "beta_note": "Explicación de la beta aplicada (si se aplicó ajuste sectorial o de la compañía).",
    "interest_expense_note": "Nota explicativa sobre los gastos por intereses financieros."
}}
"""

    max_retries = 3
    base_delay = 2

    for model_name in GEMINI_MODELS:
        for attempt in range(1, max_retries + 1):
            try:
                client = genai.Client(api_key=api_key)

                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    )
                )

                text = response.text
                if not text:
                    print(f"⚠️ Respuesta vacía de {model_name}, reintentando...")
                    continue

                text = text.strip()
                if text.startswith('```json'):
                    text = text[7:]
                elif text.startswith('```'):
                    text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]

                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    text = match.group(0)

                data = json.loads(text)

                # Validación de campos críticos
                if not isinstance(data.get("agr_list"), list) or not isinstance(data.get("opm_list"), list):
                    print("⚠️ La respuesta del LLM no contiene agr_list u opm_list como listas.")
                    continue

                # Asegurar exactamente 10 valores en cada lista
                last_agr = data['agr_list'][-1] if data['agr_list'] else rfr
                data['agr_list'] = (data['agr_list'] + [last_agr] * 10)[:10]

                last_opm = data['opm_list'][-1] if data['opm_list'] else 0.20
                data['opm_list'] = (data['opm_list'] + [last_opm] * 10)[:10]

                print(f"   ✅ Predicción LLM Extrayendo Cifras Exitosamente ({model_name}).")
                return data

            except Exception as e:
                error_str = str(e)
                print(f"⚠️ Intento {attempt}/{max_retries} con {model_name}: {error_str[:120]}")
                if attempt < max_retries:
                    time.sleep(base_delay * attempt)

    print("❌ Todos los modelos de Gemini fallaron o no respondieron adecuadamente.")
    return None


def get_damodaran_industry_with_llm(ticker, company_name, yf_industry, damodaran_industries):
    """
    Clasifica la empresa en la lista exacta de industrias de Damodaran.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key or api_key == "PEGA_AQUÍ_TU_API_KEY":
        return yf_industry

    print(f"🧠 Consultando al LLM para mapear '{ticker}' según indname.xls de Damodaran...")

    industries_list_str = "\n".join([f"- {ind}" for ind in damodaran_industries])

    prompt = f"""
Eres un analista financiero experto en la metodología de Aswath Damodaran.
Tu tarea es clasificar la empresa {company_name} (Ticker: {ticker}) en la industria correcta
según la base de datos de Damodaran (indname.xls).
La empresa ha sido clasificada por Yahoo Finance como '{yf_industry}'.

Lista exacta de las 96 industrias de Damodaran:
{industries_list_str}

Responde ÚNICAMENTE con un JSON con la estructura:
{{
    "industry": "Nombre Exacto de la Industria de Damodaran"
}}
El nombre debe ser IDÉNTICO a uno de la lista.
"""

    for model_name in GEMINI_MODELS:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            text = response.text.strip()
            if text.startswith('```json'): text = text[7:]
            elif text.startswith('```'): text = text[3:]
            if text.endswith('```'): text = text[:-3]

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: text = match.group(0)

            data = json.loads(text)
            chosen_industry = data.get("industry", yf_industry)
            if chosen_industry in damodaran_industries:
                print(f"   ✅ LLM clasificó {ticker} como: '{chosen_industry}'")
                return chosen_industry
        except Exception as e:
            print(f"⚠️ Error al consultar industria con {model_name}: {e}")

    return yf_industry
