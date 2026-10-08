import json
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

def get_llm_projections(ticker, industry, current_margins, rfr, base_revenue_growth):
    """
    Connects to Google's Gemini API to request advanced fundamental analysis 
    for 10-year AGR and OPM projections using the explicit Analyst Prompt.
    
    Usa la forma recomendada (Chat) para evitar el warning de Automatic Function Calling
    e incluye lógica de reintentos (Exponential Backoff) para manejar picos de demanda (503 UNAVAILABLE).
    """
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Intento secundario vía st.secrets (para Streamlit Cloud)
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
            
    if not api_key or api_key == "PEGA_AQUÍ_TU_API_KEY":
        print("❌ API Key no encontrada. Usando heurística.")
        return None
        
    print("🧠 Invocando al Modelo Gemini Advanced (aistudio.google.com)...")
    
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

    max_retries = 3
    base_delay = 2  # Segundos de espera inicial

    for attempt in range(1, max_retries + 1):
        try:
            client = genai.Client(api_key=api_key)
            
            # Forma recomendada: usar Chat en lugar de Models.generate_content
            chat = client.chats.create(
                model='gemini-3.6-flash',
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
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

            response = chat.send_message(full_prompt)
            text = response.text
            
            # Limpiar potencial empaquetado Markdown
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            elif text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
                
            # Buscar bloque JSON de forma segura
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
                
            data = json.loads(text)
            
            # Validación mínima de la estructura esperada
            if not isinstance(data.get("agr_list"), list) or not isinstance(data.get("opm_list"), list):
                print("⚠️ La respuesta del LLM no tiene la estructura esperada (agr_list / opm_list).")
                return None
                
            print("   ✅ Predicción LLM Extrayendo Cifras Exitosamente.")
            return data

        except Exception as e:
            error_str = str(e)
            is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str
            
            if is_unavailable and attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                print(f"⚠️️ Servidor de Gemini saturado (503). Intento {attempt}/{max_retries}. Reintentando en {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Error al contactar la API de Gemini o parsear JSON: {e}")
                return None

    return None
def get_damodaran_industry_with_llm(ticker, company_name, yf_industry, damodaran_industries):
    """
    Asks the LLM to classify the company strictly into one of Damodaran's industries.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except:
            pass
            
    if not api_key or api_key == "PEGA_AQUÍ_TU_API_KEY":
        return yf_industry
        
    print(f"🧠 Consultando al LLM para mapear '{ticker}' según indname.xls de Damodaran...")
    
    industries_list_str = "\n".join([f"- {ind}" for ind in damodaran_industries])
    
    prompt = f"""
Eres un analista financiero experto en la metodología de Aswath Damodaran. 
Tu tarea es clasificar la empresa {company_name} (Ticker: {ticker}) en la industria correcta según la base de datos exacta de Damodaran (el archivo indname.xls).
La empresa ha sido clasificada por Yahoo Finance como '{yf_industry}'.

Aquí tienes la lista exacta de las 96 industrias de Damodaran:
{industries_list_str}

Responde ÚNICAMENTE devolviendo un JSON con la estructura:
{{
    "industry": "Nombre Exacto de la Industria de Damodaran"
}}
Asegúrate de que el nombre sea IDÉNTICO letra por letra a uno de la lista provista.
"""

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            client = genai.Client(api_key=api_key)
            chat = client.chats.create(
                model='gemini-3.6-flash',
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            response = chat.send_message(prompt)
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
            else:
                print(f"   ⚠️ LLM devolvió industria no válida: '{chosen_industry}'. Usando fallback.")
                return yf_industry
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                print(f"❌ Error al consultar industria al LLM: {e}")
                return yf_industry
    
    return yf_industry
