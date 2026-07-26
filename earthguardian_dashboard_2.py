"""
EarthGuardian Live - Monitoreo Ambiental en Tiempo Real
=========================================================
Dashboard estilo "mission control" para ciudades de Panamá,
alimentado 100% por la API pública de Open-Meteo (sin API key).

Fuentes de datos:
  - Clima actual + histórico horario: https://api.open-meteo.com/v1/forecast
  - Histórico extendido (archivo):     https://archive-api.open-meteo.com/v1/archive
  - Calidad del aire (AQI europeo):    https://air-quality-api.open-meteo.com/v1/air-quality

Incluye una sección de predicción con red neuronal (MLPRegressor de
scikit-learn) entrenada con datos históricos horarios de cada ciudad.

Ejecutar con:
    pip install streamlit plotly requests pandas numpy scikit-learn
    streamlit run earthguardian_dashboard.py

--------------------------------------------------------------------
CAMBIOS EN ESTA VERSIÓN (fix de error 429 "Too Many Requests"):
  1. La sección "Mapa" ya NO hace ~20 x 2 = 40 llamadas individuales
     (una por ciudad). Ahora hace 2 llamadas en total, usando el
     soporte de Open-Meteo para múltiples coordenadas separadas por
     comas en una sola petición.
  2. Se agregó un helper `peticion_con_reintento()` que reintenta con
     backoff exponencial (1s, 2s, 4s) específicamente cuando la API
     responde 429, en vez de romper la interfaz al primer fallo.
  3. El TTL del caché del Mapa se subió a 30 min (esos datos no
     necesitan refrescarse al segundo).
--------------------------------------------------------------------
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time

print("Consultando Open-Meteo:", time.strftime("%H:%M:%S"))
# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EarthGuardian Live",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROVINCIAS = {
    "Panamá": {
        "Ciudad de Panamá": {"lat": 8.9824, "lon": -79.5199},
        "Betania":          {"lat": 8.9958, "lon": -79.5307},
        "San Miguelito":    {"lat": 9.0333, "lon": -79.5000},
        "Chepo":            {"lat": 9.1667, "lon": -79.1000},
    },
    "Panamá Oeste": {
        "La Chorrera": {"lat": 8.8800, "lon": -79.7833},
        "Arraiján":    {"lat": 8.9500, "lon": -79.6500},
        "Capira":      {"lat": 8.7500, "lon": -79.8833},
    },
    "Colón": {
        "Colón":       {"lat": 9.3547, "lon": -79.9014},
        "Portobelo":   {"lat": 9.5500, "lon": -79.6500},
    },
    "Coclé": {
        "Penonomé":    {"lat": 8.5167, "lon": -80.3583},
        "Aguadulce":   {"lat": 8.2500, "lon": -80.5500},
    },
    "Herrera": {
        "Chitré":      {"lat": 7.9667, "lon": -80.4333},
    },
    "Los Santos": {
        "Las Tablas":  {"lat": 7.7667, "lon": -80.2833},
    },
    "Veraguas": {
        "Santiago":    {"lat": 8.1000, "lon": -80.9833},
    },
    "Chiriquí": {
        "David":       {"lat": 8.4333, "lon": -82.4333},
        "Boquete":     {"lat": 8.7833, "lon": -82.4333},
    },
    "Bocas del Toro": {
        "Bocas del Toro (Isla Colón)": {"lat": 9.3400, "lon": -82.2500},
        "Changuinola":                 {"lat": 9.4333, "lon": -82.5167},
    },
    "Darién": {
        "La Palma": {"lat": 8.4064, "lon": -78.1447},
        "Metetí":   {"lat": 8.5000, "lon": -77.9500},
    },
    "Comarca Guna Yala": {
        "El Porvenir": {"lat": 9.5587, "lon": -78.9556},
    },
    "Comarca Ngäbe-Buglé": {
        "Llano Tugrí": {"lat": 8.5833, "lon": -81.7333},
    },
}

# Diccionario plano {ciudad: {lat, lon, provincia}} — útil para el mapa general
CIUDADES = {
    nombre: {**coords, "provincia": provincia}
    for provincia, ciudades in PROVINCIAS.items()
    for nombre, coords in ciudades.items()
}

# ----------------------------------------------------------------------------
# ESTILO (tema oscuro tipo "panel de control")
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1120; color: #e5e7eb; }
    .card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 8px;
    }
    .card h3 { color: #9ca3af; font-size: 13px; font-weight: 600;
               letter-spacing: 1px; margin: 0 0 6px 0; text-transform: uppercase;}
    .card .value { font-size: 32px; font-weight: 800; margin: 0; }
    .card .sub { color: #6b7280; font-size: 13px; margin-top: 4px; }
    .live-badge {
        background-color: #7f1d1d; color: #fecaca; padding: 6px 14px;
        border-radius: 20px; font-weight: 700; font-size: 13px;
    }
    .section-title { font-size: 20px; font-weight: 800; margin: 18px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HELPER: petición HTTP con reintento ante 429 (backoff exponencial)
# ----------------------------------------------------------------------------
def peticion_con_reintento(url: str, params: dict, intentos: int = 3, timeout: int = 20) -> dict:
    """
    Hace un GET y, si Open-Meteo responde 429 (Too Many Requests),
    espera con backoff exponencial (1s, 2s, 4s...) y reintenta antes
    de rendirse. Cualquier otro error HTTP se propaga de inmediato.
    """
    ultimo_error = None
    for intento in range(intentos):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                ultimo_error = requests.exceptions.HTTPError(
                    f"429 Client Error: Too Many Requests for url: {r.url}"
                )
                if intento < intentos - 1:
                    time.sleep(2 ** intento)  # 1s, 2s, 4s
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            ultimo_error = e
            if intento < intentos - 1 and "429" in str(e):
                time.sleep(2 ** intento)
                continue
            raise
    # Si se agotaron los intentos y todos fueron 429, lanza el último error
    raise ultimo_error


# ----------------------------------------------------------------------------
# LLAMADAS A LA API DE OPEN-METEO (cacheadas)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def obtener_clima(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "rain", "wind_speed_10m", "wind_direction_10m",
            "uv_index", "weather_code",
        ]),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "precipitation", "wind_speed_10m",
        ]),
        "timezone": "America/Panama",
        "forecast_days": 1,
        "past_days": 1,
    }
    return peticion_con_reintento(url, params)


@st.cache_data(ttl=300)
def obtener_calidad_aire(lat: float, lon: float) -> dict:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "european_aqi,pm2_5,pm10",
        "timezone": "America/Panama",
    }
    return peticion_con_reintento(url, params)


# --- NUEVO: versiones "batch" para la vista de Mapa -------------------------
# En vez de 1 llamada por ciudad (20 ciudades = 40 llamadas HTTP), Open-Meteo
# permite mandar varias coordenadas separadas por comas en UNA sola petición
# y devuelve una lista de resultados en el mismo orden. Esto reduce el Mapa
# de ~40 llamadas a solo 2, que es la causa principal de los 429.
@st.cache_data(ttl=1800)
def obtener_clima_multiple(coords: tuple) -> list:
    """coords: tupla de tuplas ((lat1, lon1), (lat2, lon2), ...)"""
    lats = ",".join(str(c[0]) for c in coords)
    lons = ",".join(str(c[1]) for c in coords)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "rain", "uv_index",
        ]),
        "timezone": "America/Panama",
    }
    data = peticion_con_reintento(url, params, timeout=30)
    return data if isinstance(data, list) else [data]


@st.cache_data(ttl=1800)
def obtener_calidad_aire_multiple(coords: tuple) -> list:
    lats = ",".join(str(c[0]) for c in coords)
    lons = ",".join(str(c[1]) for c in coords)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": "european_aqi",
        "timezone": "America/Panama",
    }
    data = peticion_con_reintento(url, params, timeout=30)
    return data if isinstance(data, list) else [data]


def direccion_cardinal(grados: float) -> str:
    puntos = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return puntos[round(grados / 45) % 8]


# ----------------------------------------------------------------------------
# CÁLCULO DE RIESGOS (heurísticas simples y transparentes)
# ----------------------------------------------------------------------------
def calcular_riesgos(temp, humedad, viento, lluvia, uv, aqi):
    # --- Riesgo de incendio: sube con calor/viento, baja con humedad/lluvia
    incendio = (max(0, temp - 25) * 3) + (viento * 1.2) - (humedad * 0.5) - (lluvia * 10)
    incendio = int(min(100, max(0, incendio)))

    # --- Riesgo de inundación: sube con lluvia y humedad muy alta
    inundacion = (lluvia * 15) + max(0, humedad - 80) * 2
    inundacion = int(min(100, max(0, inundacion)))

    # --- Riesgo de ola de calor: sensación térmica + humedad
    calor = max(0, temp - 28) * 6 + max(0, humedad - 60) * 0.6
    calor = int(min(100, max(0, calor)))

    # --- Riesgo de calidad del aire: directamente proporcional al AQI europeo
    aire = int(min(100, max(0, aqi * 1.4))) if aqi is not None else 20

    def nivel(valor):
        if valor < 30:
            return "BAJO", "#22c55e"
        elif valor < 60:
            return "MODERADO", "#eab308"
        elif valor < 80:
            return "ALTO", "#f97316"
        else:
            return "CRÍTICO", "#ef4444"

    return {
        "incendio": (incendio, *nivel(incendio)),
        "inundacion": (inundacion, *nivel(inundacion)),
        "calor": (calor, *nivel(calor)),
        "aire": (aire, *nivel(aire)),
    }


def generar_recomendaciones(riesgos):
    recs = []
    val, niv, _ = riesgos["incendio"]
    recs.append(("🔥", "Mantener vigilancia en áreas forestales y evitar quemas."
                 if niv in ("MODERADO", "ALTO", "CRÍTICO")
                 else "Condiciones normales, sin restricciones de quema."))
    val, niv, _ = riesgos["inundacion"]
    recs.append(("💧", "Evitar zonas bajas y quebradas, posible acumulación de agua."
                 if niv in ("ALTO", "CRÍTICO")
                 else "No se esperan inundaciones significativas."))
    val, niv, _ = riesgos["calor"]
    recs.append(("☀️", "Hidratarse constantemente y evitar exposición prolongada al sol."
                 if niv in ("MODERADO", "ALTO", "CRÍTICO")
                 else "Temperatura dentro de rangos cómodos."))
    val, niv, _ = riesgos["aire"]
    recs.append(("🍃", "Personas sensibles deben limitar actividades al aire libre."
                 if niv in ("ALTO", "CRÍTICO")
                 else "Condiciones favorables para actividades al aire libre."))
    return recs


# ----------------------------------------------------------------------------
# GAUGE CIRCULAR (plotly) — imita los anillos de la captura
# ----------------------------------------------------------------------------
def gauge_riesgo(valor: int, color: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            values=[valor, 100 - valor],
            hole=0.78,
            rotation=90,
            direction="clockwise",
            marker=dict(colors=[color, "#1f2937"]),
            textinfo="none",
            sort=False,
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=160,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{valor}/100", x=0.5, y=0.5,
                           font=dict(size=18, color="#e5e7eb"), showarrow=False)],
    )
    return fig


# ----------------------------------------------------------------------------
# PREDICCIÓN CON RED NEURONAL (versión simplificada: 1 solo modelo, sin recursión)
# ----------------------------------------------------------------------------
FEATURES = ["hora_sin", "hora_cos", "doy_sin", "doy_cos"]


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_historico(lat: float, lon: float, dias: int = 60) -> pd.DataFrame:
    """Histórico horario real de Open-Meteo, ya con las variables cíclicas listas."""
    fin = datetime.now().date() - timedelta(days=1)
    inicio = fin - timedelta(days=dias)
    data = peticion_con_reintento(
        "https://archive-api.open-meteo.com/v1/archive",
        {"latitude": lat, "longitude": lon,
         "start_date": inicio.isoformat(), "end_date": fin.isoformat(),
         "hourly": "temperature_2m", "timezone": "America/Panama"},
        timeout=30,
    )
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["hora_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365)
    return df.dropna().reset_index(drop=True)


@st.cache_resource(show_spinner="🧠 Entrenando red neuronal...")
def entrenar_modelo(lat: float, lon: float, dias: int):
    """Un solo MLP: aprende el patrón hora-del-día + día-del-año → temperatura."""
    df = obtener_historico(lat, lon, dias)
    corte = int(len(df) * 0.85)
    train, test = df.iloc[:corte], df.iloc[corte:]

    modelo = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=3000,
                     random_state=42, early_stopping=True),
    )
    modelo.fit(train[FEATURES], train["temperature_2m"])

    pred_test = modelo.predict(test[FEATURES])
    mae = mean_absolute_error(test["temperature_2m"], pred_test)
    r2 = r2_score(test["temperature_2m"], pred_test)
    limites = (df["temperature_2m"].min() - 1.5, df["temperature_2m"].max() + 1.5)
    return modelo, df, mae, r2, test, pred_test, limites


def pronosticar(modelo, ultimo_tiempo, horas: int, limites: tuple) -> pd.DataFrame:
    """Predice `horas` hacia adelante en un solo paso vectorizado (sin recursión)."""
    tiempos = [ultimo_tiempo + timedelta(hours=i) for i in range(1, horas + 1)]
    X = pd.DataFrame({
        "hora_sin": [np.sin(2 * np.pi * t.hour / 24) for t in tiempos],
        "hora_cos": [np.cos(2 * np.pi * t.hour / 24) for t in tiempos],
        "doy_sin": [np.sin(2 * np.pi * t.timetuple().tm_yday / 365) for t in tiempos],
        "doy_cos": [np.cos(2 * np.pi * t.timetuple().tm_yday / 365) for t in tiempos],
    })
    preds = np.clip(modelo.predict(X[FEATURES]), *limites)
    return pd.DataFrame({"time": tiempos, "temperatura_predicha": preds})


# ----------------------------------------------------------------------------
# PREDICCIÓN DE INCENDIOS CON RED NEURONAL (MLPClassifier)
# ----------------------------------------------------------------------------
# Fuente de eventos reales de incendio: NASA FIRMS (satélites VIIRS), gratis
# con registro en https://firms.modaps.eosdis.nasa.gov/api/map_key/
# Se busca en TODA la provincia de Panamá (bounding box amplio) para maximizar
# la cantidad de focos de calor históricos disponibles y así poder entrenar
# un clasificador real en vez de solo replicar una fórmula heurística.
# ----------------------------------------------------------------------------

# Bounding box aproximado de TODO el país de Panamá (oeste, sur, este, norte)
# Incluye desde la frontera con Costa Rica (Bocas del Toro/Chiriquí) hasta la
# frontera con Colombia (Darién), maximizando los focos de calor históricos
# disponibles para entrenar el modelo.
BBOX_PANAMA = (-83.05, 7.10, -77.15, 9.70)

FEATURES_INCENDIO = [
    "temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max",
    "vpd", "dias_sin_lluvia", "et0_fao_evapotranspiration", "shortwave_radiation_sum",
]


def calcular_vpd(temp_c: float, humedad_relativa: float) -> float:
    """Déficit de presión de vapor (kPa). Más alto = aire más 'sediento' = más riesgo."""
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (humedad_relativa / 100)
    return round(es - ea, 3)


@st.cache_data(ttl=86400, show_spinner=False)
def obtener_focos_calor_historicos(map_key: str, bbox: tuple, dias_atras: int = 365) -> pd.DataFrame:
    """
    Descarga focos de calor (incendios detectados por satélite) de NASA FIRMS
    dentro de un bounding box (west, south, east, north), hasta `dias_atras` días atrás.

    IMPORTANTE: la API de FIRMS solo acepta un máximo de 10 días por petición
    (DAY_RANGE: 1-10). Para cubrir periodos más largos (ej. 365 días), hay que
    pedirlo en bloques de 10 días usando el parámetro de fecha inicial, e ir
    acumulando los resultados.
    """
    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    # NOTA: la documentación general de FIRMS dice que el máximo es 10 días,
    # pero algunas MAP_KEY (según el tipo de cuenta) están limitadas a 5.
    # Se usa 5 para que funcione de forma consistente en todos los casos.
    BLOQUE_MAX_DIAS = 5

    hoy = datetime.now().date()
    fecha_mas_antigua = hoy - timedelta(days=dias_atras)

    partes = []
    fecha_inicio_bloque = fecha_mas_antigua
    while fecha_inicio_bloque <= hoy:
        # /api/area/csv/[MAP_KEY]/[SOURCE]/[AREA]/[DAY_RANGE]/[DATE]
        # devuelve datos desde [DATE] hasta [DATE + DAY_RANGE - 1]
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{map_key}/VIIRS_SNPP_SP/{area}/{BLOQUE_MAX_DIAS}/"
               f"{fecha_inicio_bloque.isoformat()}")
        try:
            bloque = pd.read_csv(url)
            if not bloque.empty and "acq_date" in bloque.columns:
                partes.append(bloque)
        except Exception as e:
            # Si un bloque puntual falla (ej. timeout momentáneo), no se
            # aborta todo el histórico — se sigue con el siguiente bloque.
            # Si TODOS los bloques fallan, más abajo se detecta y se avisa.
            pass

        fecha_inicio_bloque += timedelta(days=BLOQUE_MAX_DIAS)

    if not partes:
        raise RuntimeError(
            "No se pudo obtener ningún dato de NASA FIRMS. Verifica que tu "
            "MAP_KEY sea correcta y esté activa."
        )

    df = pd.concat(partes, ignore_index=True)
    if df.empty or "acq_date" not in df.columns:
        return pd.DataFrame(columns=["fecha", "latitude", "longitude"])

    df["fecha"] = pd.to_datetime(df["acq_date"]).dt.date
    df = df.drop_duplicates(subset=["fecha", "latitude", "longitude"])
    return df[["fecha", "latitude", "longitude"]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def obtener_clima_historico_incendio(lat: float, lon: float, dias_atras: int = 365) -> pd.DataFrame:
    """Histórico diario de variables relevantes para riesgo de incendio."""
    fin = datetime.now().date() - timedelta(days=1)
    inicio = fin - timedelta(days=dias_atras)
    data = peticion_con_reintento(
        "https://archive-api.open-meteo.com/v1/archive",
        {"latitude": lat, "longitude": lon,
         "start_date": inicio.isoformat(), "end_date": fin.isoformat(),
         "daily": ",".join([
             "temperature_2m_max", "relative_humidity_2m_mean",
             "wind_speed_10m_max", "precipitation_sum",
             "et0_fao_evapotranspiration", "shortwave_radiation_sum",
         ]),
         "timezone": "America/Panama"},
        timeout=30,
    )
    df = pd.DataFrame(data["daily"])
    df["fecha"] = pd.to_datetime(df["time"]).dt.date

    # Días consecutivos sin lluvia significativa (<1mm) — feature acumulativa clave
    dias_secos, contador = [], 0
    for lluvia in df["precipitation_sum"]:
        contador = contador + 1 if lluvia < 1.0 else 0
        dias_secos.append(contador)
    df["dias_sin_lluvia"] = dias_secos

    df["vpd"] = df.apply(
        lambda r: calcular_vpd(r["temperature_2m_max"], r["relative_humidity_2m_mean"]), axis=1
    )
    return df.dropna()


def construir_dataset_incendio(map_key: str, lat: float, lon: float,
                                 bbox: tuple, dias_atras: int = 365) -> tuple:
    """Une clima histórico (X) con focos de calor reales (y) por fecha."""
    clima = obtener_clima_historico_incendio(lat, lon, dias_atras)
    focos = obtener_focos_calor_historicos(map_key, bbox, dias_atras)

    fechas_con_fuego = set(focos["fecha"]) if not focos.empty else set()
    clima["hubo_incendio"] = clima["fecha"].apply(lambda f: 1 if f in fechas_con_fuego else 0)
    n_positivos = int(clima["hubo_incendio"].sum())
    return clima, n_positivos, len(fechas_con_fuego)


@st.cache_resource(show_spinner=False)
def entrenar_modelo_incendio(map_key: str, lat: float, lon: float,
                               bbox: tuple, dias_atras: int = 365, min_positivos: int = 5):
    """
    Entrena un MLPClassifier real con eventos de incendio detectados por
    satélite (NASA FIRMS) como etiqueta. Si no hay suficientes eventos
    positivos en el histórico disponible, regresa modelo=None para que la
    interfaz caiga de vuelta a la fórmula heurística en vez de mostrar un
    modelo entrenado con casi ningún ejemplo positivo (poco confiable).
    """
    df, n_positivos, n_dias_fuego = construir_dataset_incendio(map_key, lat, lon, bbox, dias_atras)

    if n_positivos < min_positivos:
        return None, df, n_positivos, n_dias_fuego, None

    X, y = df[FEATURES_INCENDIO], df["hubo_incendio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                       random_state=42, early_stopping=True),
    )
    modelo.fit(X_train, y_train)
    reporte = classification_report(y_test, modelo.predict(X_test), output_dict=True, zero_division=0)

    return modelo, df, n_positivos, n_dias_fuego, reporte


def predecir_riesgo_incendio_hoy(modelo, temp_max, humedad, viento, vpd,
                                   dias_sin_lluvia, evapotranspiracion, radiacion) -> tuple:
    """
    Devuelve (probabilidad_final, probabilidad_cruda_modelo, fue_limitada).

    La red neuronal se entrena con muy pocos ejemplos positivos (los
    incendios detectados por satélite son eventos raros), lo que puede
    producir probabilidades extremas y mal calibradas — por ejemplo, marcar
    "crítico" incluso cuando llovió recientemente y la humedad es alta.

    Para evitar mostrar un resultado que contradice la lógica climática
    básica, se aplica un techo de sentido común: si las condiciones actuales
    son claramente húmedas (llovió hoy/ayer + humedad alta), el riesgo final
    se limita, sin importar lo que diga el modelo crudo. Esto no reemplaza al
    modelo, solo evita presentarlo como si fuera 100% confiable cuando el
    propio contexto climático lo contradice.
    """
    X_hoy = pd.DataFrame([{
        "temperature_2m_max": temp_max,
        "relative_humidity_2m_mean": humedad,
        "wind_speed_10m_max": viento,
        "vpd": vpd,
        "dias_sin_lluvia": dias_sin_lluvia,
        "et0_fao_evapotranspiration": evapotranspiracion,
        "shortwave_radiation_sum": radiacion,
    }])
    probabilidad_cruda = modelo.predict_proba(X_hoy)[0][1] * 100

    # Techo de sentido común: lluvia reciente + humedad alta = condiciones
    # físicamente poco propicias para un incendio, sin importar lo que
    # "aprendió" el modelo con pocos ejemplos.
    techo = 100
    if dias_sin_lluvia == 0 and humedad >= 85:
        techo = 25   # llovió hoy y el aire está muy húmedo
    elif dias_sin_lluvia <= 1 and humedad >= 75:
        techo = 45   # lluvia muy reciente y humedad moderada-alta

    probabilidad_final = min(probabilidad_cruda, techo)
    fue_limitada = probabilidad_final < probabilidad_cruda

    return round(probabilidad_final, 1), round(probabilidad_cruda, 1), fue_limitada


@st.cache_data(ttl=1800)
def obtener_variables_incendio_hoy(lat: float, lon: float) -> dict:
    """Variables actuales necesarias para alimentar el modelo de incendio."""
    data = peticion_con_reintento(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": lat, "longitude": lon,
         "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                    "et0_fao_evapotranspiration,shortwave_radiation,wind_speed_10m",
         "past_days": 10, "forecast_days": 1,
         "timezone": "America/Panama"},
        timeout=20,
    )
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["fecha"] = df["time"].dt.date

    lluvia_diaria = df.groupby("fecha")["precipitation"].sum()
    dias_secos = 0
    for valor in reversed(lluvia_diaria.values):
        if valor < 1.0:
            dias_secos += 1
        else:
            break

    temp_max_hoy = df["temperature_2m"].tail(24).max()
    humedad_prom_hoy = df["relative_humidity_2m"].tail(24).mean()

    return {
        "temp_max": temp_max_hoy,
        "humedad": humedad_prom_hoy,
        "viento_max": df["wind_speed_10m"].tail(24).max(),
        "vpd": calcular_vpd(temp_max_hoy, humedad_prom_hoy),
        "dias_sin_lluvia": dias_secos,
        "evapotranspiracion": df["et0_fao_evapotranspiration"].tail(24).sum(),
        "radiacion": df["shortwave_radiation"].tail(24).sum(),
    }


# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌎 EarthGuardian")
    st.caption("Monitoreo ambiental en tiempo real")
    st.divider()
    seccion = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Resumen", "🗺️ Mapa", "⚠️ Riesgos", "📈 Historial",
         "🧠 Predicción IA", "🔥 Predicción Incendios", "📄 Acerca del proyecto"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("EarthGuardian Live · Datos: Open-Meteo.com")

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
LOGO_SVG = """
<svg width="54" height="54" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
  <path d="M28 2 L52 12 V26 C52 40 42 50 28 54 C14 50 4 40 4 26 V12 Z"
        fill="#0f2540" stroke="#22c55e" stroke-width="2"/>
  <circle cx="28" cy="27" r="13" fill="#1d4ed8"/>
  <path d="M17 23c3-4 8-2 10 1s6 1 9-2M16 31c4-2 9 1 12-1s7-3 10 0"
        stroke="#22c55e" stroke-width="2" fill="none" stroke-linecap="round"/>
</svg>
"""

col_logo, col_title = st.columns([0.6, 6])
with col_logo:
    st.markdown(f"<div style='margin-top:8px;'>{LOGO_SVG}</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0;'>EARTHGUARDIAN "
        "<span style='color:#22c55e;'>LIVE</span></h1>"
        "<p style='color:#9ca3af; letter-spacing:2px; margin-top:-8px;'>"
        "MONITOREO AMBIENTAL EN TIEMPO REAL</p>",
        unsafe_allow_html=True,
    )

col_prov, col_city = st.columns([1, 1])
with col_prov:
    provincia = st.selectbox("Provincia", list(PROVINCIAS.keys()))
with col_city:
    ciudad = st.selectbox("Ciudad", list(PROVINCIAS[provincia].keys()))

lat, lon = PROVINCIAS[provincia][ciudad]["lat"], PROVINCIAS[provincia][ciudad]["lon"]

# --- Mapa de Panamá como encabezado/banner ---
fig_banner = go.Figure(go.Scattermapbox(
    lat=[lat], lon=[lon], mode="markers",
    marker=dict(size=17, color="#22c55e"),
))
fig_banner.update_layout(
    mapbox=dict(style="carto-darkmatter", zoom=5.7, center=dict(lat=8.6, lon=-80.2)),
    margin=dict(l=0, r=0, t=0, b=0), height=130,
    paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
)
st.plotly_chart(fig_banner, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# OBTENER DATOS
# ----------------------------------------------------------------------------
conectado = True
try:
    clima = obtener_clima(lat, lon)
    aire = obtener_calidad_aire(lat, lon)
except requests.exceptions.RequestException as e:
    conectado = False
    if "429" in str(e):
        st.error(
            "⏳ Open-Meteo está recibiendo demasiadas peticiones en este momento "
            "(límite temporal de uso gratuito). Espera unos segundos y presiona "
            "'Actualizar', o vuelve a intentarlo en 1-2 minutos."
        )
    else:
        st.error(f"No se pudo conectar con Open-Meteo: {e}")
    st.stop()

# Hora real que reporta la API (referencia de cuándo se leyó el dato, no un reloj en vivo)
ahora_real = pd.to_datetime(clima["current"]["time"])
hora_api = ahora_real.strftime("%d/%m/%Y %H:%M")
aire_disponible = aire["current"].get("european_aqi") is not None

# --- Fuente y hora del dato: bien visible ---
st.markdown(
    f"""
    <div style='background-color:#111827; border-left:5px solid #22c55e; border-radius:8px;
                padding:12px 20px; margin-bottom:14px; display:flex; gap:32px; align-items:center;
                flex-wrap:wrap;'>
        <span style='color:#e5e7eb; font-size:15px;'>
            📡 <strong>Fuente:</strong> Open-Meteo
        </span>
        <span style='color:#e5e7eb; font-size:15px;'>
            🕐 <strong>Última actualización:</strong> {hora_api}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Indicadores de estado ---
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"{'🟢' if conectado else '🔴'} **API Meteorológica**")
with s2:
    st.markdown(f"{'🟢' if aire_disponible else '🔴'} **Calidad del Aire**")
with s3:
    st.markdown("🟢 **Red Neuronal Disponible**")

col_clock, col_refresh, col_badge = st.columns([2.5, 1, 1])

with col_clock:
    st.markdown("<p style='color:#9ca3af; margin-bottom:0;'>Hora actual (en vivo):</p>", unsafe_allow_html=True)
    # Reloj en JavaScript puro: corre en el navegador y tickea cada segundo
    # SIN depender de que Streamlit vuelva a ejecutar el script.
    components.html(
        """
        <div id="reloj-vivo" style="font-family:'Source Sans Pro',sans-serif;
             color:#22c55e; font-weight:700; font-size:18px;"></div>
        <script>
        function actualizarRelojVivo() {
            const opciones = {hour:'2-digit', minute:'2-digit', second:'2-digit',
                               hour12:true, timeZone:'America/Panama'};
            document.getElementById('reloj-vivo').innerText =
                new Date().toLocaleTimeString('es-PA', opciones);
        }
        actualizarRelojVivo();
        setInterval(actualizarRelojVivo, 1000);
        </script>
        <style> html, body { background-color: transparent; margin:0; padding:0; } </style>
        """,
        height=30,
    )

with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Actualizar", use_container_width=True):
        obtener_clima.clear()
        obtener_calidad_aire.clear()
        st.rerun()
with col_badge:
    st.markdown("<br><span class='live-badge'>🔴 EN VIVO</span>", unsafe_allow_html=True)

st.caption(
    f"📍 {ciudad}, {provincia} · lat {lat:.4f}, lon {lon:.4f} — "
    "nota: ciudades muy cercanas entre sí (menos de ~15 km) pueden mostrar "
    "valores casi idénticos porque el modelo climático de Open-Meteo trabaja "
    "con celdas de ~11-25 km de resolución. Los datos de clima se cachean; "
    "usa el botón Actualizar para forzar una lectura nueva (el reloj de arriba sí es en vivo, segundo a segundo)."
)


actual = clima["current"]
temp = actual["temperature_2m"]
sensacion = actual["apparent_temperature"]
humedad = actual["relative_humidity_2m"]
lluvia = actual.get("rain", actual.get("precipitation", 0))
viento = actual["wind_speed_10m"]
viento_dir = direccion_cardinal(actual["wind_direction_10m"])
uv = actual["uv_index"]
aqi = aire["current"].get("european_aqi")

nivel_uv = ("Bajo" if uv < 3 else "Moderado" if uv < 6 else
            "Alto" if uv < 8 else "Muy Alto" if uv < 11 else "Extremo")
nivel_aire_txt = ("Buena" if (aqi or 0) < 40 else "Moderada" if (aqi or 0) < 80
                   else "Mala" if (aqi or 0) < 120 else "Muy mala")

riesgos = calcular_riesgos(temp, humedad, viento, lluvia, uv, aqi)

# ----------------------------------------------------------------------------
# SECCIÓN: INICIO (HOME)
# ----------------------------------------------------------------------------
if seccion == "🏠 Inicio":
    st.markdown(
        """
        <div style='text-align:center; padding: 30px 10px 10px 10px;'>
            <h1 style='margin-bottom:0;'>🌎 EARTHGUARDIAN <span style='color:#22c55e;'>LIVE</span></h1>
            <p style='color:#9ca3af; font-size:16px; letter-spacing:1px;'>
                Sistema Inteligente para el Monitoreo, Clasificación y Predicción de Riesgos Ambientales
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"{'🟢' if conectado else '🔴'} **API Meteorológica**")
    with s2:
        st.markdown(f"{'🟢' if aire_disponible else '🔴'} **Calidad del Aire**")
    with s3:
        st.markdown("🟢 **Red Neuronal Disponible**")

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 🌎 ¿Qué es?")
        st.markdown(
            "**EarthGuardian Live** es un panel de monitoreo ambiental en tiempo real para ciudades "
            "de Panamá. Muestra condiciones actuales del clima, clasifica riesgos ambientales "
            "(incendio, inundación, ola de calor, calidad del aire) y usa una red neuronal para "
            "proyectar la temperatura hacia adelante."
        )
    with col_b:
        st.markdown("### 🎯 ¿Qué problema resuelve?")
        st.markdown(
            "Reúne en un solo lugar información que normalmente está dispersa (clima, calidad del "
            "aire, riesgos) y la traduce en indicadores simples de leer — pensado para que cualquier "
            "persona, no solo un meteorólogo, entienda rápido qué está pasando y qué podría venir."
        )
    with col_c:
        st.markdown("### 🛠️ ¿Qué tecnologías usa?")
        st.markdown(
            "**Python 3** · **Streamlit** (interfaz web) · **Open-Meteo** (datos climáticos abiertos, "
            "sin API key) · **Plotly** (gráficas y mapas interactivos) · **Scikit-learn** "
            "(red neuronal MLPRegressor para la predicción)."
        )

    st.divider()
    st.info(
        "👈 Usa el menú de la izquierda para explorar el **Resumen** de condiciones actuales, el "
        "**Mapa** de riesgos, el **Historial**, la **Predicción con IA**, o conocer más en "
        "**Acerca del proyecto**."
    )

# ----------------------------------------------------------------------------
# SECCIÓN: RESUMEN
# ----------------------------------------------------------------------------
elif seccion == "📊 Resumen":
    st.markdown("<div class='section-title'>CONDICIONES ACTUALES</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    tarjetas = [
        (c1, "🌡️", "TEMPERATURA", f"{temp:.1f}°C", f"Sensación: {sensacion:.1f}°C"),
        (c2, "💧", "HUMEDAD", f"{humedad:.0f}%", f"Precipitación: {lluvia:.1f} mm"),
        (c3, "🌧️", "LLUVIA", f"{lluvia:.1f} mm", "Última hora"),
        (c4, "💨", "VIENTO", f"{viento:.0f} km/h", f"Dirección: {viento_dir}"),
        (c5, "☀️", "ÍNDICE UV", f"{uv:.0f}", nivel_uv),
        (c6, "🍃", "CALIDAD DEL AIRE", f"{aqi if aqi is not None else '—'}", nivel_aire_txt),
    ]
    for col, icono, titulo, valor, sub in tarjetas:
        with col:
            st.markdown(
                f"<div class='card'><h3>{icono} {titulo}</h3>"
                f"<p class='value'>{valor}</p><p class='sub'>{sub}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>CLASIFICACIÓN DE RIESGOS AMBIENTALES</div>", unsafe_allow_html=True)
    rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1, 1, 1, 1.1])
    etiquetas = {
        "incendio": ("🔥 RIESGO DE INCENDIO", "Condiciones secas y viento moderado"),
        "inundacion": ("💧 RIESGO DE INUNDACIÓN", "Precipitación actual y drenaje"),
        "calor": ("🌡️ RIESGO DE OLA DE CALOR", "Temperatura elevada y humedad"),
        "aire": ("💨 RIESGO DE CALIDAD DEL AIRE", "Nivel de contaminación (AQI)"),
    }
    for col, key in zip([rc1, rc2, rc3, rc4], etiquetas):
        valor, nivel, color = riesgos[key]
        titulo, desc = etiquetas[key]
        with col:
            st.markdown(f"**{titulo}**")
            st.markdown(f"<p style='color:{color}; font-weight:800; font-size:20px;'>{nivel}</p>",
                        unsafe_allow_html=True)
            st.plotly_chart(gauge_riesgo(valor, color), use_container_width=True, config={"displayModeBar": False})
            st.caption(desc)

    with rc5:
        st.markdown("**🛡️ RECOMENDACIONES**")
        for icono, texto in generar_recomendaciones(riesgos):
            st.markdown(f"{icono} {texto}")

    st.markdown("<div class='section-title'>GRÁFICAS EN TIEMPO REAL (últimas 6 horas)</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        "hora": pd.to_datetime(clima["hourly"]["time"]),
        "Temperatura (°C)": clima["hourly"]["temperature_2m"],
        "Humedad (%)": clima["hourly"]["relative_humidity_2m"],
        "Viento (km/h)": clima["hourly"]["wind_speed_10m"],
        "Lluvia (mm)": clima["hourly"]["precipitation"],
    })
    ahora_grafico = ahora_real
    df = df[(df["hora"] >= ahora_grafico - timedelta(hours=6)) & (df["hora"] <= ahora_grafico)]

    fig = go.Figure()
    colores = {"Temperatura (°C)": "#ef4444", "Humedad (%)": "#3b82f6",
               "Viento (km/h)": "#22c55e", "Lluvia (mm)": "#a855f7"}
    for col in ["Temperatura (°C)", "Humedad (%)", "Viento (km/h)", "Lluvia (mm)"]:
        fig.add_trace(go.Scatter(x=df["hora"], y=df[col], name=col,
                                  mode="lines+markers", line=dict(color=colores[col])))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h", y=1.15),
        xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=10, r=10, t=10, b=10), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# SECCIÓN: MAPA  (ahora con llamadas batcheadas: 2 peticiones en vez de ~40)
# ----------------------------------------------------------------------------
elif seccion == "🗺️ Mapa":
    st.markdown("<div class='section-title'>MAPA DE RIESGOS POR PROVINCIA</div>", unsafe_allow_html=True)
    filtro_prov = st.multiselect(
        "Filtrar por provincia", list(PROVINCIAS.keys()),
        default=list(PROVINCIAS.keys()),
    )

    ciudades_filtradas = [
        (nombre, coords) for nombre, coords in CIUDADES.items()
        if coords["provincia"] in filtro_prov
    ]

    if not ciudades_filtradas:
        st.info("Selecciona al menos una provincia para ver el mapa.")
    else:
        coords_tupla = tuple((c["lat"], c["lon"]) for _, c in ciudades_filtradas)

        try:
            with st.spinner("Consultando clima y calidad del aire para todas las ciudades..."):
                climas = obtener_clima_multiple(coords_tupla)
                aires = obtener_calidad_aire_multiple(coords_tupla)
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                st.error(
                    "⏳ Open-Meteo está limitando las peticiones en este momento. "
                    "Espera un minuto y vuelve a intentarlo."
                )
            else:
                st.error(f"No se pudo conectar con Open-Meteo: {e}")
            st.stop()

        filas = []
        for (nombre, coords), c_data, a_data in zip(ciudades_filtradas, climas, aires):
            c = c_data["current"]
            a = a_data["current"].get("european_aqi")
            r = calcular_riesgos(c["temperature_2m"], c["relative_humidity_2m"],
                                  c["wind_speed_10m"], c.get("rain", 0), c["uv_index"], a)
            promedio = int(sum(v[0] for v in r.values()) / 4)
            filas.append({"Ciudad": nombre, "Provincia": coords["provincia"],
                           "lat": coords["lat"], "lon": coords["lon"],
                           "Temp (°C)": c["temperature_2m"], "Riesgo promedio": promedio})
        df_mapa = pd.DataFrame(filas)

        fig_mapa = go.Figure(go.Scattermapbox(
            lat=df_mapa["lat"], lon=df_mapa["lon"],
            mode="markers+text",
            marker=dict(size=22, color=df_mapa["Riesgo promedio"],
                        colorscale=[[0, "#22c55e"], [0.4, "#eab308"], [0.7, "#f97316"], [1, "#ef4444"]],
                        cmin=0, cmax=100, showscale=True,
                        colorbar=dict(title="Riesgo")),
            text=df_mapa["Ciudad"], textposition="top center",
            textfont=dict(color="#e5e7eb", size=12),
        ))
        fig_mapa.update_layout(
            mapbox=dict(style="carto-darkmatter", zoom=6.3,
                         center=dict(lat=8.6, lon=-80.2)),
            margin=dict(l=0, r=0, t=0, b=0), height=560,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.dataframe(df_mapa.set_index("Ciudad"), use_container_width=True)

# ----------------------------------------------------------------------------
# SECCIÓN: RIESGOS (detalle)
# ----------------------------------------------------------------------------
elif seccion == "⚠️ Riesgos":
    st.markdown("<div class='section-title'>DETALLE DE RIESGOS AMBIENTALES</div>", unsafe_allow_html=True)
    etiquetas_detalle = {
        "incendio": "🔥 Riesgo de Incendio",
        "inundacion": "💧 Riesgo de Inundación",
        "calor": "🌡️ Riesgo de Ola de Calor",
        "aire": "💨 Riesgo de Calidad del Aire",
    }
    for key, titulo in etiquetas_detalle.items():
        valor, nivel, color = riesgos[key]
        st.markdown(f"### {titulo}")
        st.progress(valor / 100)
        st.markdown(f"Nivel: <span style='color:{color}; font-weight:700;'>{nivel}</span> — {valor}/100",
                    unsafe_allow_html=True)
        st.divider()
    st.markdown("### 🛡️ Recomendaciones")
    for icono, texto in generar_recomendaciones(riesgos):
        st.write(f"{icono} {texto}")

# ----------------------------------------------------------------------------
# SECCIÓN: HISTORIAL
# ----------------------------------------------------------------------------
elif seccion == "📈 Historial":
    st.markdown(
        "<div class='section-title'>HISTORIAL REGISTRADO HASTA LA HORA ACTUAL</div>",
        unsafe_allow_html=True
    )

    df = pd.DataFrame({
        "Hora": pd.to_datetime(clima["hourly"]["time"]),
        "Temperatura (°C)": clima["hourly"]["temperature_2m"],
        "Humedad (%)": clima["hourly"]["relative_humidity_2m"],
        "Viento (km/h)": clima["hourly"]["wind_speed_10m"],
        "Lluvia (mm)": clima["hourly"]["precipitation"],
    })

    # Hora actual entregada por la propia API
    hora_actual_api = pd.to_datetime(clima["current"]["time"])

    # Mostrar solo datos que ya ocurrieron
    df = df[df["Hora"] <= hora_actual_api]

    # Ordenar desde el registro más reciente
    df = df.sort_values("Hora", ascending=False)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Descargar CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"earthguardian_{ciudad}_{datetime.now().date()}.csv"
    )

# ----------------------------------------------------------------------------
# SECCIÓN: PREDICCIÓN CON RED NEURONAL
# ----------------------------------------------------------------------------
elif seccion == "🧠 Predicción IA":
    st.markdown("<div class='section-title'>PREDICCIÓN DE TEMPERATURA CON RED NEURONAL</div>",
                unsafe_allow_html=True)
    st.caption(
        f"Red neuronal MLP (perceptrón multicapa, 2 capas ocultas 32→16) entrenada con el histórico "
        f"real de **{ciudad}, {provincia}** vía la API de archivo de Open-Meteo."
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        dias_hist = st.slider("Días de historial para entrenar", 30, 120, 60, step=10)
        horizonte = st.radio(
            "Horizonte de predicción",
            ["🕐 Próximas 24 horas", "📅 Próximo mes (30 días)"],
        )
        horas_pred = 24 if horizonte.startswith("🕐") else 24 * 30
        if st.button("🔄 Reentrenar desde cero", use_container_width=True):
            entrenar_modelo.clear()

    try:
        with col_b:
            modelo, df, mae, r2, test, pred_test, limites = entrenar_modelo(lat, lon, dias_hist)
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            st.error(
                "⏳ Open-Meteo está limitando las peticiones en este momento. "
                "Espera un minuto y vuelve a intentarlo."
            )
        else:
            st.error(f"No se pudo descargar el histórico de Open-Meteo: {e}")
        st.stop()

    with st.spinner(f"Calculando pronóstico para las próximas {horas_pred} horas..."):
        pronostico = pronosticar(modelo, df["time"].iloc[-1], horas_pred, limites)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='card'><h3>📊 Error medio (MAE)</h3>"
                    f"<p class='value'>{mae:.2f}°C</p></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='card'><h3>🎯 R² (ajuste en validación)</h3>"
                    f"<p class='value'>{r2:.3f}</p></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card'><h3>🌡️ Rango histórico real</h3>"
                    f"<p class='value'>{limites[0]+1.5:.1f}° – {limites[1]-1.5:.1f}°C</p></div>",
                    unsafe_allow_html=True)

    calidad_r2 = ("excelente" if r2 >= 0.9 else "buena" if r2 >= 0.7 else
                  "aceptable" if r2 >= 0.5 else "débil")
    with st.expander("❓ ¿Qué significan el MAE y el R²?"):
        st.markdown(
            f"**MAE (Error Absoluto Medio)** — en promedio, cada predicción del modelo se equivoca "
            f"por **±{mae:.2f}°C** respecto a la temperatura real. Mientras más bajo, mejor: un MAE "
            f"de 1°C es muy bueno para temperatura; uno de 4-5°C ya es un margen de error considerable.\n\n"
            f"**R² (coeficiente de determinación)** — indica qué tan bien el modelo explica el "
            f"comportamiento real de la temperatura, en una escala de 0 a 1. Un R² de **{r2:.3f}** "
            f"significa que el modelo captura aproximadamente el **{max(r2, 0)*100:.0f}%** del patrón "
            f"(el resto es variación que el modelo no logra explicar, como frentes fríos o lluvias "
            f"puntuales). En este caso, el ajuste es **{calidad_r2}**.\n\n"
            f"En resumen: entre más bajo el MAE y más cercano a 1 el R², más confiable es la predicción."
        )

    fig = go.Figure()
    ultimos = df.tail(7 * 24)
    fig.add_trace(go.Scatter(x=ultimos["time"], y=ultimos["temperature_2m"],
                              name="Temperatura real (histórico)", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=test["time"], y=pred_test,
                              name="Predicción en validación", line=dict(color="#eab308", dash="dot")))

    if horas_pred <= 24:
        fig.add_trace(go.Scatter(x=pronostico["time"], y=pronostico["temperatura_predicha"],
                                  name="Pronóstico próximas 24h", line=dict(color="#22c55e", dash="dash")))
    else:
        # Para el horizonte de 1 mes se agrega por día (promedio, mín, máx) — más legible y honesto
        diario = pronostico.copy()
        diario["fecha"] = diario["time"].dt.date
        resumen = diario.groupby("fecha")["temperatura_predicha"].agg(["mean", "min", "max"]).reset_index()
        fig.add_trace(go.Scatter(x=resumen["fecha"], y=resumen["mean"],
                                  name="Promedio diario (patrón estacional)", line=dict(color="#22c55e", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(resumen["fecha"]) + list(resumen["fecha"][::-1]),
            y=list(resumen["max"]) + list(resumen["min"][::-1]),
            fill="toself", fillcolor="rgba(34,197,94,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Rango diario esperado", showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h", y=1.15),
        xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937", title="°C"),
        margin=dict(l=10, r=10, t=10, b=10), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Pronóstico detallado")
    if horas_pred <= 24:
        tabla = pronostico.copy()
        tabla["Hora"] = tabla["time"].dt.strftime("%d/%m %H:%M")
        tabla["Temperatura predicha (°C)"] = tabla["temperatura_predicha"].round(1)
        st.dataframe(tabla[["Hora", "Temperatura predicha (°C)"]], use_container_width=True, hide_index=True)
    else:
        tabla = resumen.copy()
        tabla["Fecha"] = pd.to_datetime(tabla["fecha"]).dt.strftime("%d/%m/%Y")
        tabla["Promedio (°C)"] = tabla["mean"].round(1)
        tabla["Mínima (°C)"] = tabla["min"].round(1)
        tabla["Máxima (°C)"] = tabla["max"].round(1)
        st.dataframe(tabla[["Fecha", "Promedio (°C)", "Mínima (°C)", "Máxima (°C)"]],
                     use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Modelo educativo (versión simplificada): una sola red neuronal que aprende únicamente el "
        "patrón hora-del-día + día-del-año (climatología), sin usar la temperatura actual como punto de "
        "partida. Es honesto y estable en cualquier horizonte, pero el pronóstico de 24h no reflejará "
        "anomalías del momento (ej. un frente frío pasajero hoy). Para 1 mes, recuerda que ningún modelo "
        "del mundo —ni GFS ni ECMWF— predice con precisión más allá de ~10-16 días: esto muestra el "
        "patrón estacional típico, no el clima exacto de cada día."
    )

# ----------------------------------------------------------------------------
# SECCIÓN: PREDICCIÓN DE INCENDIOS CON RED NEURONAL (MLPClassifier + FIRMS)
# ----------------------------------------------------------------------------
elif seccion == "🔥 Predicción Incendios":
    st.markdown("<div class='section-title'>PREDICCIÓN DE INCENDIOS CON RED NEURONAL</div>",
                unsafe_allow_html=True)
    st.caption(
        "Clasificador MLP (red neuronal) entrenado con focos de calor reales detectados por "
        "satélite (NASA FIRMS) en TODO el territorio de Panamá, combinados con variables "
        "climáticas históricas de Open-Meteo."
    )

    with st.expander("🔑 ¿Cómo obtengo mi MAP_KEY gratis de NASA FIRMS?"):
        st.markdown(
            "1. Entra a **https://firms.modaps.eosdis.nasa.gov/api/map_key/**\n"
            "2. Ingresa tu correo electrónico (no necesitas contraseña ni cuenta compleja)\n"
            "3. Copia el código que te generan y pégalo aquí abajo\n\n"
            "Es gratis, sin límite de uso significativo para este proyecto."
        )

    map_key = st.text_input("🔑 Tu MAP_KEY de NASA FIRMS", type="password",
                             placeholder="Pega aquí tu clave...")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        dias_hist_fuego = st.slider("Días de historial a analizar", 60, 365, 365, step=30)
        if st.button("🔄 Reentrenar desde cero", use_container_width=True, key="reentrenar_fuego"):
            entrenar_modelo_incendio.clear()

    if not map_key:
        st.info(
            "👆 Ingresa tu MAP_KEY de NASA FIRMS para entrenar el modelo con incendios reales. "
            "Mientras tanto, puedes seguir usando la estimación de riesgo de incendio en la "
            "sección **⚠️ Riesgos**, basada en una fórmula (no en una red neuronal entrenada)."
        )
        st.stop()

    try:
        with col_b:
            with st.spinner("🧠 Descargando focos de calor históricos (en bloques de 5 días) "
                             "y entrenando red neuronal... esto puede tardar 40-80 segundos."):
                modelo_fuego, df_fuego, n_positivos, n_dias_fuego, reporte = entrenar_modelo_incendio(
                    map_key, lat, lon, BBOX_PANAMA, dias_hist_fuego
                )
    except RuntimeError as e:
        st.error(f"No se pudo conectar con NASA FIRMS: {e}")
        st.info("Verifica que tu MAP_KEY sea correcta y esté activa.")
        st.stop()

    if modelo_fuego is None:
        st.warning(
            f"⚠️ Solo se detectaron **{n_positivos} días con incendio** en los últimos "
            f"{dias_hist_fuego} días en TODO el territorio de Panamá — muy pocos ejemplos para "
            f"entrenar una red neuronal confiable (se necesitan al menos 5). "
            f"Prueba ampliando el rango de días arriba, o vuelve más adelante cuando exista "
            f"más histórico acumulado.\n\n"
            f"Mientras tanto, usa la sección **⚠️ Riesgos**, que sí funciona con la fórmula "
            f"heurística de incendio."
        )
        st.stop()

    # --- Métricas del modelo ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='card'><h3>🔥 Días con incendio detectado</h3>"
                    f"<p class='value'>{n_positivos}</p>"
                    f"<p class='sub'>de {len(df_fuego)} días analizados</p></div>",
                    unsafe_allow_html=True)
    with m2:
        precision = reporte.get("1", {}).get("precision", 0) * 100
        st.markdown(f"<div class='card'><h3>🎯 Precisión (clase incendio)</h3>"
                    f"<p class='value'>{precision:.0f}%</p></div>", unsafe_allow_html=True)
    with m3:
        recall = reporte.get("1", {}).get("recall", 0) * 100
        st.markdown(f"<div class='card'><h3>📡 Sensibilidad (recall)</h3>"
                    f"<p class='value'>{recall:.0f}%</p></div>", unsafe_allow_html=True)

    with st.expander("❓ ¿Qué significan estas métricas?"):
        st.markdown(
            "**Precisión** — de todos los días que el modelo marcó como \"riesgo de incendio\", "
            "qué porcentaje realmente tuvo un foco de calor detectado por satélite.\n\n"
            "**Sensibilidad (recall)** — de todos los días que SÍ tuvieron un incendio real, "
            "qué porcentaje logró detectar el modelo. En seguridad ambiental, un recall alto "
            "suele ser más importante que la precisión: es preferible una alerta de más que "
            "un incendio real sin avisar.\n\n"
            "Como los incendios son eventos poco frecuentes, estas métricas pueden variar "
            "bastante según cuántos días de historial uses."
        )

    st.divider()

    # --- Predicción para HOY con las condiciones actuales ---
    st.markdown("### 🔮 Riesgo de incendio hoy")
    try:
        vars_hoy = obtener_variables_incendio_hoy(lat, lon)
        prob_hoy, prob_cruda, fue_limitada = predecir_riesgo_incendio_hoy(
            modelo_fuego,
            vars_hoy["temp_max"], vars_hoy["humedad"], vars_hoy["viento_max"],
            vars_hoy["vpd"], vars_hoy["dias_sin_lluvia"],
            vars_hoy["evapotranspiracion"], vars_hoy["radiacion"],
        )
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo calcular el riesgo de hoy: {e}")
        st.stop()

    if fue_limitada:
        st.info(
            f"ℹ️ La red neuronal calculó un riesgo crudo de **{prob_cruda}%**, pero las "
            f"condiciones actuales (lluvia reciente + humedad alta) hacen ese número poco "
            f"realista físicamente. Se muestra un valor ajustado de **{prob_hoy}%** por sentido "
            f"común climático. Esto suele pasar cuando el modelo se entrenó con muy pocos "
            f"incendios reales — mientras más historial acumules, más confiable será la RNA sola."
        )

    color_riesgo = ("#22c55e" if prob_hoy < 30 else "#eab308" if prob_hoy < 60
                     else "#f97316" if prob_hoy < 80 else "#ef4444")
    nivel_riesgo = ("BAJO" if prob_hoy < 30 else "MODERADO" if prob_hoy < 60
                     else "ALTO" if prob_hoy < 80 else "CRÍTICO")

    col_gauge, col_vars = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(gauge_riesgo(int(prob_hoy), color_riesgo), use_container_width=True,
                         config={"displayModeBar": False})
        st.markdown(f"<p style='text-align:center; color:{color_riesgo}; font-weight:800; "
                    f"font-size:20px;'>{nivel_riesgo}</p>", unsafe_allow_html=True)
    with col_vars:
        st.markdown(f"- 🌡️ Temperatura máxima (24h): **{vars_hoy['temp_max']:.1f}°C**")
        st.markdown(f"- 💧 Humedad relativa promedio: **{vars_hoy['humedad']:.0f}%**")
        st.markdown(f"- 💨 Viento máximo (24h): **{vars_hoy['viento_max']:.0f} km/h**")
        st.markdown(f"- 🏜️ Días consecutivos sin lluvia: **{vars_hoy['dias_sin_lluvia']}** "
                    f"(0 = llovió hoy)")
        st.markdown(f"- 📉 Déficit de presión de vapor (VPD): **{vars_hoy['vpd']:.2f} kPa**")

    st.divider()

    # --- Mapa de focos de calor históricos (todo el país) ---
    if n_dias_fuego > 0:
        st.markdown("### 🗺️ Focos de calor detectados en Panamá (histórico)")
        focos_mapa = obtener_focos_calor_historicos(map_key, BBOX_PANAMA, dias_hist_fuego)
        if not focos_mapa.empty:
            fig_focos = go.Figure()
            fig_focos.add_trace(go.Scattermapbox(
                lat=focos_mapa["latitude"], lon=focos_mapa["longitude"],
                mode="markers", name="Focos de calor",
                marker=dict(size=7, color="#ef4444", opacity=0.6),
            ))
            # Resalta la ciudad seleccionada para dar contexto de referencia
            fig_focos.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lon], mode="markers", name=f"{ciudad} (seleccionada)",
                marker=dict(size=16, color="#22c55e"),
            ))
            fig_focos.update_layout(
                mapbox=dict(style="carto-darkmatter", zoom=6.2,
                             center=dict(lat=8.6, lon=-80.2)),
                margin=dict(l=0, r=0, t=0, b=0), height=460,
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.05, font=dict(color="#e5e7eb")),
            )
            st.plotly_chart(fig_focos, use_container_width=True)
            st.caption(f"📍 Total de focos de calor detectados en el país: **{len(focos_mapa)}**")

    st.caption(
        "⚠️ Modelo educativo: entrenado con datos reales de focos de calor satelitales (NASA FIRMS) "
        "de TODO el territorio de Panamá, ya que un área tan pequeña como una sola ciudad rara vez "
        "tiene suficientes eventos históricos para entrenar un clasificador confiable. La predicción "
        "de hoy usa las condiciones climáticas locales de la ciudad seleccionada. Los focos de calor "
        "satelitales pueden incluir quemas agrícolas controladas, no solo incendios forestales "
        "descontrolados."
    )

# ----------------------------------------------------------------------------
# SECCIÓN: ACERCA DEL PROYECTO
# ----------------------------------------------------------------------------
elif seccion == "📄 Acerca del proyecto":
    st.markdown("<div class='section-title'>ACERCA DEL PROYECTO</div>", unsafe_allow_html=True)

    st.markdown("### 🎯 Objetivo")
    st.markdown(
        "EarthGuardian Live busca ofrecer **monitoreo ambiental accesible y en tiempo real** para "
        "ciudades de Panamá, combinando datos meteorológicos abiertos con un modelo de inteligencia "
        "artificial que ayuda a anticipar patrones de temperatura. La meta es que cualquier persona "
        "—estudiantes, docentes, o el público general— pueda consultar condiciones actuales, riesgos "
        "ambientales (incendio, inundación, ola de calor, calidad del aire) y una proyección de "
        "temperatura, sin depender de servicios de pago."
    )

    st.markdown("### 🛠️ Tecnologías utilizadas")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            "- **Python 3** — lenguaje base del proyecto\n"
            "- **Streamlit** — framework del dashboard web\n"
            "- **Plotly** — gráficas y mapas interactivos\n"
            "- **Pandas / NumPy** — procesamiento de datos"
        )
    with col_t2:
        st.markdown(
            "- **scikit-learn (MLPRegressor)** — red neuronal de predicción\n"
            "- **Requests** — consumo de APIs REST\n"
            "- **Streamlit Cloud** — despliegue y hosting\n"
            "- **HTML/CSS/JavaScript** — reloj en vivo y estilos"
        )

    st.markdown("### 🔌 APIs")
    st.markdown(
        "- **[Open-Meteo Forecast API](https://open-meteo.com/)** — clima actual y horario, sin API key\n"
        "- **Open-Meteo Archive API** — histórico horario real, usado para entrenar la red neuronal\n"
        "- **Open-Meteo Air Quality API** — índice de calidad del aire (AQI europeo)"
    )

    st.markdown("### 🏗️ Arquitectura")
    st.code(
        "   Open-Meteo API\n"
        "         │\n"
        "         ▼\n"
        "     EarthGuardian\n"
        "         │\n"
        "     Python + Streamlit\n"
        "         │\n"
        "         ▼\n"
        "     Dashboard Web",
        language=None,
    )
    st.markdown(
        "En términos simples: el navegador del usuario carga el **Dashboard Web**, que corre sobre "
        "**Python + Streamlit**. Ese backend le pide datos en tiempo real a la **API de Open-Meteo** "
        "(clima, histórico y calidad del aire), los procesa —clasifica riesgos, entrena la red "
        "neuronal con el histórico— y renderiza el resultado como tarjetas, gráficas y mapas."
    )

    st.markdown("### ✍️ Autores")
    st.markdown("**Luz Alba Andrade**")

st.markdown(
    """
    <div style='text-align:center; padding:28px 0 10px 0; margin-top:20px;
                border-top:1px solid #1f2937; color:#9ca3af;'>
        <p style='font-size:18px; font-weight:800; color:#e5e7eb; margin-bottom:2px;'>
            🌎 EarthGuardian Live
        </p>
        <p style='margin:2px 0; font-size:14px;'>
            Sistema Inteligente para el Monitoreo,<br>Clasificación y Predicción de Riesgos Ambientales
        </p>
        <p style='margin-top:10px; font-size:13px; color:#6b7280;'>
            Python • Streamlit • Open-Meteo • Plotly • Scikit-learn
        </p>
        <p style='margin-top:10px; font-size:11px; color:#4b5563;'>
            Universidad / Learning Vila · Datos proporcionados por Open-Meteo.com · sin necesidad de API key
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
