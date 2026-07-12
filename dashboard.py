import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="EarthGuardian Live",
    page_icon="🌎",
    layout="wide"
)

CIUDADES = {
    "Panamá": {"lat": 8.9833, "lon": -79.5167},
    "Colón": {"lat": 9.3592, "lon": -79.9014},
    "David": {"lat": 8.4333, "lon": -82.4333},
    "Santiago": {"lat": 8.1000, "lon": -80.9833},
    "Boquete": {"lat": 8.7802, "lon": -82.4414},
}


# =========================
# FUNCIONES API OPEN-METEO
# =========================

def consultar_clima(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "pressure_msl,"
            "uv_index"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "timezone": "auto",
        "forecast_days": 1
    }

    respuesta = requests.get(url, params=params, timeout=10)
    respuesta.raise_for_status()
    return respuesta.json()


def consultar_calidad_aire(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "european_aqi,"
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "timezone": "auto"
    }

    respuesta = requests.get(url, params=params, timeout=10)
    respuesta.raise_for_status()
    return respuesta.json()


# =========================
# MOTOR DE RIESGOS
# =========================

def limitar(valor):
    return max(0, min(100, int(valor)))


def riesgo_incendio(temp, humedad, viento, lluvia):
    riesgo = 0

    if temp >= 34:
        riesgo += 35
    elif temp >= 30:
        riesgo += 20

    if humedad <= 35:
        riesgo += 35
    elif humedad <= 50:
        riesgo += 20

    if viento >= 30:
        riesgo += 20
    elif viento >= 15:
        riesgo += 10

    if lluvia <= 0:
        riesgo += 10

    return limitar(riesgo)


def riesgo_inundacion(lluvia, humedad):
    riesgo = 0

    if lluvia >= 20:
        riesgo += 60
    elif lluvia >= 10:
        riesgo += 35
    elif lluvia >= 3:
        riesgo += 15

    if humedad >= 90:
        riesgo += 25
    elif humedad >= 80:
        riesgo += 10

    return limitar(riesgo)


def riesgo_ola_calor(temp, uv, humedad):
    riesgo = 0

    if temp >= 35:
        riesgo += 45
    elif temp >= 31:
        riesgo += 30

    if uv >= 8:
        riesgo += 30
    elif uv >= 6:
        riesgo += 15

    if humedad >= 80:
        riesgo += 20

    return limitar(riesgo)


def riesgo_calidad_aire(aqi):
    if aqi is None:
        return 0
    return limitar(aqi)


def clasificar(valor):
    if valor <= 25:
        return "🟢 BAJO"
    elif valor <= 50:
        return "🟡 MODERADO"
    elif valor <= 75:
        return "🟠 ALTO"
    else:
        return "🔴 CRÍTICO"


def recomendacion(r_incendio, r_inundacion, r_calor, r_aire):
    recomendaciones = []

    if r_incendio > 50:
        recomendaciones.append("🔥 Evitar quemas y vigilar zonas forestales.")

    if r_inundacion > 50:
        recomendaciones.append("💧 Revisar drenajes y zonas cercanas a ríos.")

    if r_calor > 50:
        recomendaciones.append("☀️ Evitar exposición prolongada al sol.")

    if r_aire > 50:
        recomendaciones.append("🌫️ Reducir actividades al aire libre.")

    if not recomendaciones:
        recomendaciones.append("✅ Condiciones ambientales dentro de rango aceptable.")

    return recomendaciones


# =========================
# INTERFAZ
# =========================

st.title("🌎 EarthGuardian Live")
st.subheader("Dashboard de Monitoreo Ambiental en Tiempo Real")
st.caption("Datos climáticos y de calidad del aire obtenidos desde Open-Meteo.")

ciudad = st.sidebar.selectbox(
    "Selecciona ciudad",
    list(CIUDADES.keys())
)

lat = CIUDADES[ciudad]["lat"]
lon = CIUDADES[ciudad]["lon"]

st.sidebar.write("Fuente de datos:")
st.sidebar.write("✅ Open-Meteo Weather API")
st.sidebar.write("✅ Open-Meteo Air Quality API")
st.sidebar.write("✅ Sin API Key")

try:
    clima = consultar_clima(lat, lon)
    aire = consultar_calidad_aire(lat, lon)

    actual = clima["current"]
    actual_aire = aire["current"]

    temperatura = actual.get("temperature_2m", 0)
    humedad = actual.get("relative_humidity_2m", 0)
    lluvia = actual.get("precipitation", 0)
    viento = actual.get("wind_speed_10m", 0)
    presion = actual.get("pressure_msl", 0)
    uv = actual.get("uv_index", 0)

    aqi = actual_aire.get("european_aqi", 0)
    pm10 = actual_aire.get("pm10", 0)
    pm25 = actual_aire.get("pm2_5", 0)
    co = actual_aire.get("carbon_monoxide", 0)
    no2 = actual_aire.get("nitrogen_dioxide", 0)
    so2 = actual_aire.get("sulphur_dioxide", 0)
    ozone = actual_aire.get("ozone", 0)

    r_incendio = riesgo_incendio(temperatura, humedad, viento, lluvia)
    r_inundacion = riesgo_inundacion(lluvia, humedad)
    r_calor = riesgo_ola_calor(temperatura, uv, humedad)
    r_aire = riesgo_calidad_aire(aqi)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🌡 Temperatura", f"{temperatura} °C")
    col2.metric("💧 Humedad", f"{humedad} %")
    col3.metric("🌧 Lluvia", f"{lluvia} mm")
    col4.metric("🌬 Viento", f"{viento} km/h")
    col5.metric("☀ Índice UV", f"{uv}")

    st.divider()

    st.header("Clasificación de Riesgos Ambientales")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🔥 Riesgo de Incendio", clasificar(r_incendio), f"{r_incendio}/100")
    c2.metric("💧 Riesgo de Inundación", clasificar(r_inundacion), f"{r_inundacion}/100")
    c3.metric("☀ Riesgo de Ola de Calor", clasificar(r_calor), f"{r_calor}/100")
    c4.metric("🌫 Calidad del Aire", clasificar(r_aire), f"{r_aire}/100")

    st.divider()

    st.header("Calidad del Aire")

    a1, a2, a3, a4 = st.columns(4)

    a1.metric("AQI", aqi)
    a2.metric("PM2.5", pm25)
    a3.metric("PM10", pm10)
    a4.metric("CO", co)

    st.write("NO₂:", no2)
    st.write("SO₂:", so2)
    st.write("O₃:", ozone)

    st.divider()

    st.header("Recomendaciones")

    for rec in recomendacion(r_incendio, r_inundacion, r_calor, r_aire):
        st.warning(rec)

    st.divider()

    st.header("Gráficas en Tiempo Real")

    hourly = clima["hourly"]

    df = pd.DataFrame({
        "hora": hourly["time"],
        "temperatura": hourly["temperature_2m"],
        "humedad": hourly["relative_humidity_2m"],
        "lluvia": hourly["precipitation"],
        "viento": hourly["wind_speed_10m"]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["hora"],
        y=df["temperatura"],
        mode="lines+markers",
        name="Temperatura"
    ))

    fig.add_trace(go.Scatter(
        x=df["hora"],
        y=df["humedad"],
        mode="lines+markers",
        name="Humedad"
    ))

    fig.add_trace(go.Scatter(
        x=df["hora"],
        y=df["viento"],
        mode="lines+markers",
        name="Viento"
    ))

    fig.add_trace(go.Scatter(
        x=df["hora"],
        y=df["lluvia"],
        mode="lines+markers",
        name="Lluvia"
    ))

    fig.update_layout(
        title="Variables ambientales del día",
        xaxis_title="Hora",
        yaxis_title="Valor",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error("No se pudo consultar la API.")
    st.write(e)