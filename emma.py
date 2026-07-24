import requests
import streamlit as st

st.set_page_config(
    page_title="EcoMonitor",
    page_icon="🌎",
    layout="wide"
)

# Coordenadas de la Provincia de Panamá
LAT = 8.9833
LON = -79.5167


@st.cache_data(ttl=300)
def consultar_clima():

    clima = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "hourly": "uv_index",
            "timezone": "America/Panama",
            "forecast_days": 1
        }
    ).json()

    aire = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT,
            "longitude": LON,
            "current": "european_aqi",
            "timezone": "America/Panama"
        }
    ).json()

    return clima, aire


def clasificar_temperatura(temp):

    if temp < 20:
        return "🟢 Baja"

    elif temp < 30:
        return "🟡 Normal"

    elif temp < 35:
        return "🟠 Alta"

    else:
        return "🔴 Crítica"


st.title("🌎 EcoMonitor")
st.subheader("Provincia de Panamá")

clima, aire = consultar_clima()

actual = clima["current"]

temperatura = actual["temperature_2m"]
humedad = actual["relative_humidity_2m"]
lluvia = actual["precipitation"]
viento = actual["wind_speed_10m"]

hora = actual["time"]

uv = clima["hourly"]["uv_index"][0]

aqi = aire["current"]["european_aqi"]

c1, c2, c3 = st.columns(3)

c1.metric("🌡 Temperatura", f"{temperatura} °C")
c2.metric("💧 Humedad", f"{humedad}%")
c3.metric("🌧 Lluvia", f"{lluvia} mm")

c4, c5, c6 = st.columns(3)

c4.metric("💨 Viento", f"{viento} km/h")
c5.metric("☀ Índice UV", uv)
c6.metric("🌫 Calidad del Aire", aqi)

st.divider()

st.header("Clasificación")

st.success(
    f"Temperatura: {clasificar_temperatura(temperatura)}"
)

st.info("Humedad: ____________________")
st.info("Lluvia: ____________________")
st.info("Viento: ____________________")
st.info("Índice UV: ____________________")
st.info("Calidad del aire: ____________________")

st.caption(f"Última actualización: {hora}")