import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI # Alternativa OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import io # Para manejar el archivo subido en memoria
from datetime import datetime # Import para la hora
import pytz # Import para la zona horaria


# --- Configuración Inicial ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Error: API Key no encontrada...")
    st.stop()
else:
    # ----> AÑADE ESTA LÍNEA PARA DEBUG <----
    st.sidebar.warning(f"DEBUG: Usando Key que termina en: ...{API_KEY[-6:]}") # Muestra últimos 6 chars en la sidebar

# Configura el LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY,
                                 temperature=0.5, convert_system_message_to_human=True)
    st.sidebar.success("DEBUG: Objeto LLM inicializado OK.")
except Exception as e:
    st.error(f"Error al inicializar el LLM: {e}")
    st.stop()


# --- Carga de Reglas Operativas desde Archivo JSON ---
RULES_FILENAME = "reglas.json" # Nombre del archivo de reglas JSON
OPERATIONAL_RULES = {} # Inicializa un diccionario vacío

try:
    # Construye la ruta completa al archivo (más robusto)
    # __file__ se refiere a este script (app.py)
    script_dir = os.path.dirname(__file__)
    rules_path = os.path.join(script_dir, RULES_FILENAME)

    with open(rules_path, 'r', encoding='utf-8') as f:
        OPERATIONAL_RULES = json.load(f) # Carga el contenido JSON en el diccionario
    st.sidebar.info(f"Reglas cargadas correctamente desde '{RULES_FILENAME}'.")

except FileNotFoundError:
    st.error(f"Error Crítico: No se encontró el archivo de reglas '{RULES_FILENAME}' en la ruta: {rules_path}")
    st.info(f"Asegúrate de que el archivo '{RULES_FILENAME}' exista en la misma carpeta que app.py.")
    st.stop() # Detiene la ejecución si no hay reglas
except json.JSONDecodeError as e:
    st.error(f"Error Crítico: El archivo '{RULES_FILENAME}' no contiene JSON válido.")
    st.error(f"Detalle del error: {e}")
    st.info("Por favor, verifica la sintaxis del archivo JSON.")
    st.stop() # Detiene la ejecución si las reglas están mal formateadas
except Exception as e:
    st.error(f"Error inesperado al cargar las reglas desde '{RULES_FILENAME}': {e}")
    st.stop()


# --- Funciones Auxiliares ---
def check_rules(data_row):
    """Revisa una fila de datos contra las reglas definidas (cargadas desde archivo).
       Adaptado para usar SensorID y columnas calculadas del CSV de ejemplo."""
    anomalies_found = []
    if not isinstance(OPERATIONAL_RULES, dict):
        st.warning("Las reglas operacionales no se cargaron correctamente.")
        return anomalies_found

    # Extraer datos relevantes de la fila del CSV (que ahora es un diccionario)
    sensor_id = data_row.get('SensorID')
    parameter_col = data_row.get('Parameter') # Columna Parameter original
    value_col = data_row.get('Value')
    timestamp = data_row.get('Timestamp', 'N/A')
    # Extraer columnas calculadas específicas
    calc_sh_k_col = data_row.get('Calculated_SH_K')

    for rule_name, rule_details in OPERATIONAL_RULES.items():
        rule_param_target = rule_details.get("parameter") # El SensorID o nombre de columna que busca la regla
        if not rule_param_target: # Saltar si la regla no tiene parámetro definido
             continue

        value_to_check = None
        current_sensor_id_or_calc = None # Para saber qué estamos comparando

        # --- Lógica para seleccionar el valor correcto del CSV ---
        if rule_param_target == "Calculated_SH_K":
            # La regla busca específicamente el SH calculado
            value_to_check = pd.to_numeric(calc_sh_k_col, errors='coerce')
            current_sensor_id_or_calc = "Calculated_SH_K"
        elif sensor_id == rule_param_target:
             # La regla busca un SensorID específico, usamos el valor de la columna 'Value'
             value_to_check = pd.to_numeric(value_col, errors='coerce')
             current_sensor_id_or_calc = sensor_id
        else:
            # La regla no coincide con el SensorID de esta fila ni con una columna calculada conocida
            continue # Pasar a la siguiente regla

        # --- Si encontramos un valor relevante, aplicar la regla ---
        if pd.notna(value_to_check):
            threshold = rule_details.get("threshold")
            condition = rule_details.get("condition")
            condition_met = False

            # Evaluar la condición
            if condition == "greater_than" and isinstance(threshold, (int, float)) and value_to_check > threshold:
                condition_met = True
            elif condition == "less_than" and isinstance(threshold, (int, float)) and value_to_check < threshold:
                condition_met = True
            # Añadir más condiciones si es necesario

            if condition_met:
                # Formatear la descripción y recomendación
                description_template = rule_details.get("anomaly_description", "Anomalía detectada en {parameter}.")
                recommendation_hint = rule_details.get("recommendation_hint", "Revisar sistema.")
                severity = rule_details.get("severity", "Indeterminada")

                try:
                    # Usar .format() para insertar valores en la descripción
                    description = description_template.format(parameter=current_sensor_id_or_calc, value=value_to_check)
                except KeyError:
                    description = description_template # Usar plantilla si falla el formato

                anomalies_found.append({
                    "rule_name": rule_name,
                    "timestamp": timestamp,
                    "equipment_id": current_sensor_id_or_calc, # Usamos el ID/Nombre que coincidió
                    "parameter": current_sensor_id_or_calc, # Parámetro que violó la regla
                    "value": value_to_check, # Valor que violó
                    "threshold": threshold,
                    "description": description,
                    "recommendation_hint": recommendation_hint,
                    "severity": severity
                })
    return anomalies_found

def format_anomalies_for_llm(anomalies):
    """Formatea la lista de anomalías para el prompt del LLM."""
    if not anomalies:
        return "No se detectaron anomalías significativas.", "No aplica."

    formatted_text = "Se detectaron las siguientes anomalías:\n"
    recommendation_hints = []
    severities = []

    # Ordenar por severidad (Alta > Media > Baja) y luego por timestamp si existe
    severity_order = {"Alta": 0, "Media": 1, "Baja": 2}
    try:
      anomalies.sort(key=lambda x: (severity_order.get(x.get("severity"), 99), x.get("timestamp", "")))
    except TypeError:
      # Fallback si timestamp no es comparable (ej. NaT)
      anomalies.sort(key=lambda x: severity_order.get(x.get("severity"), 99))


    for anomaly in anomalies:
        ts = anomaly.get('timestamp', 'N/A')
        # Asegurarse que ts sea string para el formato
        ts_str = str(ts) if pd.notna(ts) else 'N/A'
        desc = anomaly.get('description', 'Descripción no disponible')
        sev = anomaly.get('severity', 'N/A')
        formatted_text += f"- [{ts_str}] {desc} (Severidad: {sev})\n"

        hint = anomaly.get('recommendation_hint')
        rule_name = anomaly.get('rule_name')
        hint_text = f"- {hint} (Relacionado con regla: {rule_name})"
        if hint and hint_text not in recommendation_hints:
             recommendation_hints.append(hint_text)
        if sev:
             severities.append(sev)

    highest_severity = "Baja"
    if "Alta" in severities:
        highest_severity = "Alta"
    elif "Media" in severities:
        highest_severity = "Media"

    formatted_recommendations = "\nSugerencias de revisión basadas en reglas:\n" + "\n".join(recommendation_hints)
    formatted_text += f"\nNivel de severidad general detectado: {highest_severity}"

    return formatted_text, formatted_recommendations


# --- Prompt para el LLM ---
prompt_template = """
Eres un asistente experto en operaciones y seguridad de plantas de refrigeración industrial por amoníaco (NH3), enfocado en normativas y buenas prácticas como las del IIAR. Trabajas para una planta frigorífica en Chile (Ubicación Actual: {location}, Hora Actual: {current_time}).

Se ha ejecutado un análisis automático de los datos recientes de sensores (cargados desde un archivo CSV) y se encontraron las siguientes condiciones operativas anómalas basadas en las reglas definidas en el archivo 'reglas.json':

**Resumen de Anomalías Detectadas:**
{anomalies_summary}

**Sugerencias Preliminares Basadas en Reglas (del archivo reglas.json):**
{recommendation_hints}

**Tu Tarea:**
Basándote **estrictamente** en la información proporcionada sobre las anomalías detectadas y las sugerencias preliminares:
1.  Genera un **informe conciso y claro** para el equipo de operaciones y mantenimiento.
2.  Describe la situación general observada en los datos y el **nivel de riesgo potencial** (considerando la severidad indicada).
3.  Proporciona un **plan de acción claro y priorizado**, comenzando por las anomalías de mayor severidad. Si hay múltiples anomalías, sugiere un orden lógico de revisión o si pueden estar relacionadas.
4.  **Enfatiza siempre la importancia de la seguridad** al trabajar con amoníaco (NH3) y la necesidad de investigar estas desviaciones según los protocolos específicos de la planta antes de realizar ajustes mayores.
5.  Mantén un tono profesional, técnico y directo. No inventes información que no esté presente en las anomalías o sugerencias proporcionadas. No añadas saludos genéricos al inicio o final.

**Informe y Plan de Acción:**
"""

# Obtener ubicación y hora actual (añadido para contexto en el prompt)
from datetime import datetime
import pytz # Necesita instalarse: pip install pytz

try:
    # Usar la zona horaria de Chile Continental
    chile_tz = pytz.timezone('Chile/Continental')
    current_dt_chile = datetime.now(chile_tz)
    current_time_str = current_dt_chile.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    current_location = "El Monte, RM, Chile" # Placeholder, podría ser más dinámico
except Exception:
    current_time_str = "No disponible"
    current_location = "No disponible"


PROMPT = PromptTemplate(template=prompt_template, input_variables=["anomalies_summary", "recommendation_hints"],
                        partial_variables={"location": current_location, "current_time": current_time_str}) # Añadir variables parciales
llm_chain = LLMChain(prompt=PROMPT, llm=llm)


# --- Interfaz de Streamlit ---

st.title("❄️ Asistente IA para Operación de Sistemas de Refrigeración NH3")
st.subheader("Prototipo de Decisión Basada en Datos para IIAR (Reglas desde Archivo JSON)")
st.markdown(f"""
Esta aplicación analiza datos de sensores de un archivo CSV, los compara con las reglas operativas cargadas desde `{RULES_FILENAME}`,
identifica anomalías y utiliza IA para generar un informe y plan de acción.
(Ubicación: {current_location} | Hora Actual: {current_time_str})
""")

st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV con datos de sensores", type="csv")

if uploaded_file is not None:
    try:
        # Leer el archivo CSV subido en memoria
        # Especificar la codificación puede ser importante
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        # Intentar convertir Timestamp a datetime si existe y manejar errores
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

        st.sidebar.success(f"Archivo '{uploaded_file.name}' cargado con {len(data)} filas.")

        st.subheader("1. Datos de Sensores Cargados (Vista Previa)")
        st.dataframe(data.head())

        st.subheader("2. Reglas Operativas Aplicadas")
        with st.expander(f"Ver Reglas Cargadas desde '{RULES_FILENAME}'"):
            st.json(OPERATIONAL_RULES) # Muestra las reglas leídas del archivo

        st.subheader("3. Análisis de Anomalías")
        all_anomalies = []
        # Iterar por cada fila del dataframe para aplicar las reglas
        for index, row in data.iterrows():
            # Pasar la fila como un diccionario para que .get() funcione bien
            anomalies_in_row = check_rules(row.to_dict())
            if anomalies_in_row:
                all_anomalies.extend(anomalies_in_row)

        if not all_anomalies:
            st.success("✅ No se detectaron anomalías según las reglas definidas en los datos cargados.")
        else:
            st.warning(f"🚨 Se detectaron {len(all_anomalies)} anomalías!")
            # Crear un DataFrame para mostrar las anomalías de forma ordenada
            anomalies_df = pd.DataFrame(all_anomalies)
            # Preparar para ordenar por severidad y timestamp
            severity_order = {"Alta": 0, "Media": 1, "Baja": 2}
            anomalies_df['severity_order'] = anomalies_df['severity'].map(severity_order)
            # Asegurarse que timestamp es comparable, si no, usar un valor por defecto
            anomalies_df['timestamp_sort'] = pd.to_datetime(anomalies_df['timestamp'], errors='coerce').fillna(pd.Timestamp.min)
            anomalies_df = anomalies_df.sort_values(by=['severity_order', 'timestamp_sort'])

            # Seleccionar y mostrar columnas relevantes
            display_cols = ['timestamp', 'equipment_id', 'parameter', 'value', 'threshold', 'severity', 'description']
            st.dataframe(anomalies_df[display_cols])

            st.subheader("4. Generación de Informe y Plan de Acción con IA")
            # Formatear para el LLM
            anomalies_summary_text, recommendation_hints_text = format_anomalies_for_llm(all_anomalies)

            with st.spinner("Generando informe y plan de acción con IA..."):
                try:
                    # Ejecutar la cadena LLM
                    response = llm_chain.invoke({
                        "anomalies_summary": anomalies_summary_text,
                        "recommendation_hints": recommendation_hints_text
                    })
                    st.markdown("**Informe y Plan de Acción Generado por IA:**")
                    st.markdown(response['text']) # El resultado de la cadena LLM
                except Exception as e:
                    st.error(f"Error al generar la respuesta del LLM: {e}")
                    st.exception(e) # Muestra el traceback para depuración

    except Exception as e:
        st.error(f"Error al procesar el archivo CSV: {e}")
        st.exception(e)
else:
    st.info("Por favor, carga un archivo CSV con datos de sensores para iniciar el análisis.")


# --- Sidebar con Notas Adicionales ---
st.sidebar.markdown("---")
st.sidebar.header("Notas Importantes")
st.sidebar.markdown(f"""
- Las reglas operativas se cargan desde `{RULES_FILENAME}`. Modifica ese archivo para cambiar los umbrales o añadir reglas.
- El archivo CSV debe contener columnas como `Timestamp`, `Equipment`, `Parameter`, `Value`.
- Asegúrate de tener tu API Key (`GOOGLE_API_KEY` o `OPENAI_API_KEY`) configurada en `.env` o `st.secrets`.
- Este es un prototipo. La precisión de las recomendaciones depende de la calidad de las reglas en `{RULES_FILENAME}` y la capacidad del LLM. **Siempre verificar con criterio experto antes de actuar.**
""")
st.sidebar.header("Próximos Pasos (Futuro)")
st.sidebar.markdown("""
- Conectar a una base de datos real para reglas.
- Integración con SCADA en tiempo real.
- Implementar **aprendizaje**: Usar RAG con embeddings (FAISS/ChromaDB) y documentos (manuales, SOPs, históricos) para responder preguntas más complejas o identificar patrones no cubiertos por reglas fijas.
- Añadir visualizaciones de datos (gráficos de tendencia).
""")