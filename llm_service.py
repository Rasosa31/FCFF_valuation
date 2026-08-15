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
Tu única misión es producir proyecciones de ingresos (revenues) y Margen Operativo (Operating Margin) realista, 
coherentes y bien fundamentados para alimentar la app de valoración FCFF del usuario. Nunca inventes números sin 
justificación explícita.

**Compañía a valorar:** {ticker}
**Industria / Sector:** {industry}
**Margen Operativo Promedio Histórico:** {current_margins:.2%}
**Risk Free Rate (Crecimiento Terminal esperado):** {rfr:.2%}
**Crecimiento Consenso de Analistas (Año 1 prospectivo):** {base_revenue_growth:.2%}

### Reglas para Crecimiento de Ingresos (AGR):
1. **Etapa del ciclo de vida:** Clasifica a la empresa según Damodaran (Startup / High Growth / Mature Growth / Mature Stable / Decline) justificando con evidencia.
2. **Análisis de TAM/SOM:** Estima qué porción del mercado es capturable a 10 años evaluando barreras de entrada.
3. **Distribución del CAGR:** Construye un arreglo de 10 años. Si es Front-loaded, decrece rápido. El año 10 debe acercarse asintóticamente a la Tasa Libre de Riesgo ({rfr:.2%}).

### Reglas para Margen Operativo (OPM):
1. El Agente anterior lo mantenía estático. Ahora debes proyectarlo.
2. ¿Aumentará por economías de escala / madurez? ¿Descenderá por competencia y caducidad tecnológica?
3. Evalúa promedios empíricos históricos ({current_margins:.2%}) contra márgenes diana de su sector industrial. Construye un recorrido de 10 años.

Debes devolver OBLIGATORIAMENTE un JSON que sea programacionalmente parseable, con la siguiente estructura exacta:
{{
    "revenue_narrative": "Resumen ejecutivo de la narrativa de crecimiento de 2-4 párrafos",
    "agr_list": [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.05, 0.046, 0.046, 0.046],
    "margin_narrative": "Resumen ejecutivo argumentando la evolución de los márgenes",
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
