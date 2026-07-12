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
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
# LLAMADAS A LA API DE OPEN-METEO (cacheadas 5 min)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
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
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def obtener_calidad_aire(lat: float, lon: float) -> dict:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "european_aqi,pm2_5,pm10",
        "timezone": "America/Panama",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


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
# PREDICCIÓN CON RED NEURONAL (MLPRegressor)
# ----------------------------------------------------------------------------
FEATURES_MODELO = [
    "hora_sin", "hora_cos", "doy_sin", "doy_cos",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag24",
    "humedad_lag1", "viento_lag1",
]
FEATURES_ESTACIONALES = ["hora_sin", "hora_cos", "doy_sin", "doy_cos"]


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_historico(lat: float, lon: float, dias: int = 60) -> pd.DataFrame:
    """Descarga histórico horario real desde la API de archivo de Open-Meteo."""
    fin = datetime.now().date() - timedelta(days=1)   # el archivo suele tener 1-2 días de rezago
    inicio = fin - timedelta(days=dias)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": inicio.isoformat(),
        "end_date": fin.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "America/Panama",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.dropna().reset_index(drop=True)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables cíclicas de tiempo + rezagos (lags) para el modelo."""
    df = df.sort_values("time").reset_index(drop=True).copy()
    df["hora_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365)
    for lag in (1, 2, 3, 24):
        df[f"temp_lag{lag}"] = df["temperature_2m"].shift(lag)
    df["humedad_lag1"] = df["relative_humidity_2m"].shift(1)
    df["viento_lag1"] = df["wind_speed_10m"].shift(1)
    return df.dropna().reset_index(drop=True)


@st.cache_resource(show_spinner="🧠 Entrenando red neuronal con datos históricos...")
def entrenar_modelo(lat: float, lon: float, dias: int):
    """
    Entrena DOS redes neuronales sobre el mismo histórico:
      - `modelo`            (con rezagos): para el pronóstico recursivo de 24h.
      - `modelo_estacional`  (solo hora/día del año): para el patrón de 1 mes,
        sin recursión, así que no puede "irse de rango" acumulando error.
    """
    df_hist = obtener_historico(lat, lon, dias)
    df_feat = construir_features(df_hist)

    corte = int(len(df_feat) * 0.85)
    train, test = df_feat.iloc[:corte], df_feat.iloc[corte:]

    modelo = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=3000,
            random_state=42,
            early_stopping=True,
        ),
    )
    modelo.fit(train[FEATURES_MODELO], train["temperature_2m"])

    pred_test = modelo.predict(test[FEATURES_MODELO])
    mae = mean_absolute_error(test["temperature_2m"], pred_test)
    r2 = r2_score(test["temperature_2m"], pred_test)

    modelo_estacional = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=3000,
            random_state=42,
            early_stopping=True,
        ),
    )
    modelo_estacional.fit(df_feat[FEATURES_ESTACIONALES], df_feat["temperature_2m"])

    # Límites de seguridad: nunca predecir fuera del rango histórico real (+margen pequeño)
    temp_min = float(df_feat["temperature_2m"].min())
    temp_max = float(df_feat["temperature_2m"].max())
    limites = (temp_min - 1.5, temp_max + 1.5)

    return modelo, modelo_estacional, df_feat, mae, r2, test, pred_test, limites


def pronosticar(modelo, df_feat: pd.DataFrame, horas: int, limites: tuple) -> pd.DataFrame:
    """Pronóstico recursivo hora a hora (24h), con límites de seguridad basados en el histórico real."""
    ventana_temp = df_feat["temperature_2m"].tolist()[-24:]
    humedad_actual = df_feat["relative_humidity_2m"].iloc[-1]
    viento_actual = df_feat["wind_speed_10m"].iloc[-1]
    tiempo_actual = df_feat["time"].iloc[-1]
    minimo, maximo = limites

    filas = []
    for i in range(1, horas + 1):
        t = tiempo_actual + timedelta(hours=i)
        fila = {
            "hora_sin": np.sin(2 * np.pi * t.hour / 24),
            "hora_cos": np.cos(2 * np.pi * t.hour / 24),
            "doy_sin": np.sin(2 * np.pi * t.timetuple().tm_yday / 365),
            "doy_cos": np.cos(2 * np.pi * t.timetuple().tm_yday / 365),
            "temp_lag1": ventana_temp[-1],
            "temp_lag2": ventana_temp[-2],
            "temp_lag3": ventana_temp[-3],
            "temp_lag24": ventana_temp[-24] if len(ventana_temp) >= 24 else ventana_temp[0],
            "humedad_lag1": humedad_actual,
            "viento_lag1": viento_actual,
        }
        pred = float(modelo.predict(pd.DataFrame([fila])[FEATURES_MODELO])[0])
        pred = min(max(pred, minimo), maximo)   # nunca se sale del rango histórico real
        filas.append({"time": t, "temperatura_predicha": pred})
        ventana_temp.append(pred)

    return pd.DataFrame(filas)


def pronosticar_estacional(modelo_estacional, df_feat: pd.DataFrame, horas: int, limites: tuple) -> pd.DataFrame:
    """
    Pronóstico climatológico para horizontes largos (ej. 1 mes): SIN recursión.
    Cada hora se calcula de forma independiente solo a partir de su hora del día
    y día del año, así que no hay forma de que el error se acumule o "dispare".
    """
    tiempo_actual = df_feat["time"].iloc[-1]
    minimo, maximo = limites
    tiempos = [tiempo_actual + timedelta(hours=i) for i in range(1, horas + 1)]

    X = pd.DataFrame({
        "hora_sin": [np.sin(2 * np.pi * t.hour / 24) for t in tiempos],
        "hora_cos": [np.cos(2 * np.pi * t.hour / 24) for t in tiempos],
        "doy_sin": [np.sin(2 * np.pi * t.timetuple().tm_yday / 365) for t in tiempos],
        "doy_cos": [np.cos(2 * np.pi * t.timetuple().tm_yday / 365) for t in tiempos],
    })
    preds = modelo_estacional.predict(X[FEATURES_ESTACIONALES])
    preds = np.clip(preds, minimo, maximo)

    return pd.DataFrame({"time": tiempos, "temperatura_predicha": preds})


# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌎 EarthGuardian")
    st.caption("Monitoreo ambiental en tiempo real")
    st.divider()
    seccion = st.radio(
        "Navegación",
        ["📊 Resumen", "🗺️ Mapa", "⚠️ Riesgos", "📈 Historial", "🧠 Predicción IA"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("EarthGuardian Live · Datos: Open-Meteo.com")

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
col_title, col_prov, col_city, col_time, col_badge = st.columns([2.6, 1.2, 1.3, 1.3, 0.8])
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0;'>EARTHGUARDIAN "
        "<span style='color:#22c55e;'>LIVE</span></h1>"
        "<p style='color:#9ca3af; letter-spacing:2px; margin-top:-8px;'>"
        "MONITOREO AMBIENTAL EN TIEMPO REAL</p>",
        unsafe_allow_html=True,
    )
with col_prov:
    provincia = st.selectbox("Provincia", list(PROVINCIAS.keys()))
with col_city:
    ciudad = st.selectbox("Ciudad", list(PROVINCIAS[provincia].keys()))
with col_time:
    st.markdown(
        f"<p style='color:#9ca3af; margin-bottom:0;'>Actualización:</p>"
        f"<p style='color:#22c55e; font-weight:700; font-size:20px;'>"
        f"{datetime.now().strftime('%I:%M:%S %p')}</p>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br><span class='live-badge'>🔴 EN VIVO</span>", unsafe_allow_html=True)

lat, lon = PROVINCIAS[provincia][ciudad]["lat"], PROVINCIAS[provincia][ciudad]["lon"]
st.caption(
    f"📍 {ciudad}, {provincia} · lat {lat:.4f}, lon {lon:.4f} — "
    "nota: ciudades muy cercanas entre sí (menos de ~15 km) pueden mostrar "
    "valores casi idénticos porque el modelo climático de Open-Meteo trabaja "
    "con celdas de ~11-25 km de resolución."
)

# ----------------------------------------------------------------------------
# OBTENER DATOS
# ----------------------------------------------------------------------------
try:
    clima = obtener_clima(lat, lon)
    aire = obtener_calidad_aire(lat, lon)
except requests.exceptions.RequestException as e:
    st.error(f"No se pudo conectar con Open-Meteo: {e}")
    st.stop()

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
# SECCIÓN: RESUMEN
# ----------------------------------------------------------------------------
if seccion == "📊 Resumen":
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

    st.markdown("<div class='section-title'>GRÁFICAS EN TIEMPO REAL (últimas 24 horas)</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        "hora": pd.to_datetime(clima["hourly"]["time"]),
        "Temperatura (°C)": clima["hourly"]["temperature_2m"],
        "Humedad (%)": clima["hourly"]["relative_humidity_2m"],
        "Viento (km/h)": clima["hourly"]["wind_speed_10m"],
        "Lluvia (mm)": clima["hourly"]["precipitation"],
    })
    ahora = datetime.now()
    df = df[(df["hora"] >= ahora - timedelta(hours=6)) & (df["hora"] <= ahora + timedelta(hours=1))]

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
# SECCIÓN: MAPA
# ----------------------------------------------------------------------------
elif seccion == "🗺️ Mapa":
    st.markdown("<div class='section-title'>MAPA DE RIESGOS POR PROVINCIA</div>", unsafe_allow_html=True)
    filtro_prov = st.multiselect(
        "Filtrar por provincia", list(PROVINCIAS.keys()),
        default=list(PROVINCIAS.keys()),
    )
    filas = []
    for nombre, coords in CIUDADES.items():
        if coords["provincia"] not in filtro_prov:
            continue
        c = obtener_clima(coords["lat"], coords["lon"])["current"]
        a = obtener_calidad_aire(coords["lat"], coords["lon"])["current"].get("european_aqi")
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
    st.markdown("<div class='section-title'>HISTORIAL DE 24 HORAS</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        "Hora": pd.to_datetime(clima["hourly"]["time"]),
        "Temperatura (°C)": clima["hourly"]["temperature_2m"],
        "Humedad (%)": clima["hourly"]["relative_humidity_2m"],
        "Viento (km/h)": clima["hourly"]["wind_speed_10m"],
        "Lluvia (mm)": clima["hourly"]["precipitation"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                        file_name=f"earthguardian_{ciudad}_{datetime.now().date()}.csv")

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
            (modelo, modelo_estacional, df_feat, mae, r2,
             test, pred_test, limites) = entrenar_modelo(lat, lon, dias_hist)
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo descargar el histórico de Open-Meteo: {e}")
        st.stop()

    with st.spinner(f"Calculando pronóstico para las próximas {horas_pred} horas..."):
        if horas_pred <= 24:
            pronostico = pronosticar(modelo, df_feat, horas_pred, limites)
        else:
            pronostico = pronosticar_estacional(modelo_estacional, df_feat, horas_pred, limites)

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

    fig = go.Figure()
    ultimos = df_feat.tail(7 * 24)
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

    if horas_pred <= 24:
        st.caption(
            "⚠️ Modelo educativo: MLP entrenado con temperatura, humedad y viento históricos, más "
            "variables cíclicas de hora/día del año. El pronóstico es recursivo (cada hora predicha "
            "alimenta la siguiente), así que el error tiende a crecer con el horizonte. No reemplaza "
            "modelos meteorológicos profesionales (GFS/ICON)."
        )
    else:
        st.caption(
            "⚠️ Importante sobre el pronóstico a 1 mes: ningún modelo meteorológico del mundo — "
            "ni siquiera los profesionales como GFS o ECMWF — tiene capacidad predictiva real más "
            "allá de ~10-16 días; el clima es un sistema caótico. Lo que ves aquí para semanas 3-4 "
            "es esencialmente el **patrón estacional aprendido** por la red (climatología: qué tan "
            "cálido/húmedo suele ser ese día del año en esta zona), no una predicción específica día "
            "a día. Útil para tendencias generales, no para decisiones puntuales."
        )

st.caption("🌎 EarthGuardian Live · Datos proporcionados por Open-Meteo.com (sin necesidad de API key)")
