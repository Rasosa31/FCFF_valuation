# FCFF_valuation

Estructura limpia de archivos esenciales (12 archivos)

Archivo	                    Función / Propósito
app.py	                    Aplicación web interactiva en Streamlit.
ai_agent.py	                Orquestador del Agente de Valoración Autónomo.
valuation_engine.py	        Motor de valoración FCFF, cálculo de WACC y simulación Monte Carlo.
llm_service.py	            Integración con Gemini para el perfil cualitativo y     
                            proyecciones dinámicas.
yfinance_service.py	        Extracción y normalización de estados financieros vía Yahoo Finance.
damodaran_service.py	    Integración con NYU Stern: Implied ERP, Sales-to-Capital y mapeo de 
                            industrias.
sec_edgar_service.py	    Extracción de Stock Options (BSM) y contingencia de deuda en SEC 
                            EDGAR.
indname_cache.csv	        Base de datos local optimizada para mapeo de ~48,000 empresas a 
                            Damodaran.
damodaran_industries.json	Lista canónica de las 96 industrias de Damodaran.
requirements.txt	        Dependencias y librerías requeridas del proyecto.
README.md	                Documentación del repositorio.
.gitignore	                Reglas de exclusión de Git.