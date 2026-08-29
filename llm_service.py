import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def get_llm_projections(ticker, industry, current_margins, rfr, base_revenue_growth):
    """
    Connects to Google's Gemini API to request advanced fundamental analysis 
    for 10-year AGR and OPM projections using the explicit Analyst Prompt.
    """
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "PEGA_AQUÍ_TU_API_KEY":
        return None
        
    print("🧠 Invocando al Modelo Gemini Advanced (aistudio.google.com)...")
    try:
        client = genai.Client(api_key=api_key)
        
        full_prompt = f"""
Eres un analista de valoración fundamental de élite especializado en el método FCFF (Free Cash Flow to Firm).
Tu única misión es proyectar ingresos (tasa de crecimiento anual) y margen operacional de forma realista y no lineal, 
basándote exclusivamente en el análisis profundo de toda la información disponible (sector, ciclo económico, reinversión, 
competencia, tendencias históricas y perspectivas futuras) para alimentar la app de valoración FCFF del usuario. 
Nunca inventes números sin justificación explícita.

**Compañía a valorar:** {ticker}
**Industria / Sector:** {industry}
**Margen Operativo Promedio Histórico:** {current_margins:.2%}
**Risk Free Rate (Crecimiento Terminal esperado):** {rfr:.2%}
**Crecimiento Consenso de Analistas (Año 1 prospectivo):** {base_revenue_growth:.2%}

### Reglas para Crecimiento de Ingresos (AGR):
1. La curva de crecimiento puede acelerar, desacelerar, estabilizarse o invertirse en cualquier momento del horizonte (ej. alto crecimiento inicial y luego moderación, o lo contrario). 
2. PROHÍBE cualquier decrecimiento lineal automático. 
3. El año 10 debe converger racionalmente hacia una tasa estable, típicamente acercándose a la Tasa Libre de Riesgo ({rfr:.2%}).

### Reglas para Margen Operativo (OPM):
1. Evoluciona dinámicamente según reinversión, eficiencia, pricing power y salud del negocio. 
2. Puede mejorar, deteriorarse o estabilizarse. 
3. PROHÍBE el uso de promedio histórico estático de forma plana; debes proyectar su evolución en un horizonte de 10 años.

Las proyecciones deben reflejar las condiciones reales de la empresa y su entorno. Justifica brevemente la forma de cada curva en las narrativas.

Debes devolver OBLIGATORIAMENTE un JSON que sea programacionalmente parseable, con la siguiente estructura exacta:
{{
    "revenue_narrative": "Justificación de 2-4 párrafos explicando por qué la curva de ingresos acelera/desacelera.",
    "agr_list": [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.05, 0.046, 0.046, 0.046],
    "margin_narrative": "Justificación de la evolución dinámica del margen operativo (eficiencia, pricing power, etc).",
    "opm_list": [0.45, 0.46, 0.47, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49]
}}
"""
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        text = response.text
        data = json.loads(text)
        print("   ✅ Predicción LLM Extrayendo Cifras Exitosamente.")
        return data
    except Exception as e:
        print(f"❌ Error al contactar la API de Gemini: {e}")
        return None
