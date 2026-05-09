from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import VARIABLES_NUMERICAS
from src.prediccion import cargar_sistema, predecir_productividad, ranking_equipos

st.set_page_config(
    page_title="Simulador Productividad Harvester",
    page_icon="🌲",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando modelo de productividad...")
def cargar_modelo_cache():
    return cargar_sistema(auto_entrenar=True)


modelo, metadata = cargar_modelo_cache()
opciones = metadata["opciones_categoricas"]
defaults = metadata["numeric_defaults"]
mins = metadata["numeric_min"]
maxs = metadata["numeric_max"]
clima_defaults = metadata["clima_defaults"]

st.title("🌲 Simulador de productividad para equipos Harvester")
st.caption(
    "Aplicación basada en el notebook RegresionLinealHV_v9. Estima productividad M3/HORA "
    "y compara equipos bajo las mismas condiciones operativas."
)

with st.expander("Cómo interpretar esta herramienta", expanded=False):
    st.write(
        "El modelo no dice que un equipo sea universalmente mejor. Compara equipos dentro de un escenario "
        "controlado: misma zona, especie, suelo, turno, carga operativa y clima. La salida es un apoyo "
        "analítico para planeación, no una regla causal absoluta."
    )

st.sidebar.header("Condiciones del escenario")

modo = st.sidebar.radio(
    "Tipo de simulación",
    ["Ranking de equipos", "Predicción de un equipo"],
    index=0,
)

# Campos categóricos recomendados por negocio
marca_opciones = ["TODAS"] + [x for x in opciones["MARCA"] if x != "OTROS"] + ["OTROS"]
marca_filtro = st.sidebar.selectbox("Marca de máquina", marca_opciones)

if modo == "Predicción de un equipo":
    equipo = st.sidebar.selectbox("Equipo Harvester", opciones["EQUIPO"])
else:
    equipo = opciones["EQUIPO"][0]

zona = st.sidebar.selectbox("Zona", opciones["ZONA"])
contratista = st.sidebar.selectbox("Contratista", opciones["CONTRATISTA"])
especie = st.sidebar.selectbox("Especie", opciones["ESPECIE"])
turno = st.sidebar.selectbox("Turno", opciones["TURNO"])
suelo = st.sidebar.selectbox("Suelo", opciones["SUELO"])

st.sidebar.header("Variables operativas")

pendiente = st.sidebar.number_input(
    "Pendiente promedio finca",
    min_value=float(mins["PENDIENTE PROMEDIO FINCA"]),
    max_value=float(maxs["PENDIENTE PROMEDIO FINCA"]),
    value=float(defaults["PENDIENTE PROMEDIO FINCA"]),
    step=1.0,
)

total_arboles = st.sidebar.number_input(
    "Total de árboles",
    min_value=float(mins["TOTAL DE ARBOLES"]),
    max_value=float(maxs["TOTAL DE ARBOLES"]),
    value=float(defaults["TOTAL DE ARBOLES"]),
    step=10.0,
)

diametro = st.sidebar.number_input(
    "Diámetro promedio",
    min_value=float(mins["DIAMETRO"]),
    max_value=float(maxs["DIAMETRO"]),
    value=float(defaults["DIAMETRO"]),
    step=0.5,
)

tiempo_programado = st.sidebar.number_input(
    "Tiempo programado",
    min_value=float(mins["T PROGRAMADO"]),
    max_value=float(maxs["T PROGRAMADO"]),
    value=float(defaults["T PROGRAMADO"]),
    step=0.5,
)

horas_paradas = st.sidebar.number_input(
    "Horas de otras paradas",
    min_value=float(mins["HORAS DE OTRAS PARADA"]),
    max_value=float(maxs["HORAS DE OTRAS PARADA"]),
    value=float(defaults["HORAS DE OTRAS PARADA"]),
    step=0.25,
)

st.sidebar.header("Clima")
tipo_clima = st.sidebar.selectbox("Tipo de clima", ["SECO", "LLUVIOSO", "MANUAL"])

if tipo_clima in ["SECO", "LLUVIOSO"]:
    clima = clima_defaults[tipo_clima]
    temp_prom = clima["clima_temp_promedio_dia_c"]
    temp_min = clima["clima_temp_min_dia_c"]
    temp_max = clima["clima_temp_max_dia_c"]
    precipitacion = clima["clima_precipitacion_dia_mm"]
    viento = clima["clima_viento_promedio_dia_kmh"]
    st.sidebar.info("Se usan valores climáticos típicos del histórico para el tipo seleccionado.")
else:
    temp_prom = st.sidebar.number_input(
        "Temperatura promedio (°C)",
        min_value=float(mins["clima_temp_promedio_dia_c"]),
        max_value=float(maxs["clima_temp_promedio_dia_c"]),
        value=float(defaults["clima_temp_promedio_dia_c"]),
        step=0.5,
    )
    temp_min = st.sidebar.number_input(
        "Temperatura mínima (°C)",
        min_value=float(mins["clima_temp_min_dia_c"]),
        max_value=float(maxs["clima_temp_min_dia_c"]),
        value=float(defaults["clima_temp_min_dia_c"]),
        step=0.5,
    )
    temp_max = st.sidebar.number_input(
        "Temperatura máxima (°C)",
        min_value=float(mins["clima_temp_max_dia_c"]),
        max_value=float(maxs["clima_temp_max_dia_c"]),
        value=float(defaults["clima_temp_max_dia_c"]),
        step=0.5,
    )
    precipitacion = st.sidebar.number_input(
        "Precipitación diaria (mm)",
        min_value=float(mins["clima_precipitacion_dia_mm"]),
        max_value=float(maxs["clima_precipitacion_dia_mm"]),
        value=float(defaults["clima_precipitacion_dia_mm"]),
        step=0.5,
    )
    viento = st.sidebar.number_input(
        "Viento promedio (km/h)",
        min_value=float(mins["clima_viento_promedio_dia_kmh"]),
        max_value=float(maxs["clima_viento_promedio_dia_kmh"]),
        value=float(defaults["clima_viento_promedio_dia_kmh"]),
        step=0.5,
    )

escenario = {
    "EQUIPO": equipo,
    "MARCA": marca_filtro if marca_filtro != "TODAS" else opciones["MARCA"][0],
    "ZONA": zona,
    "CONTRATISTA": contratista,
    "ESPECIE": especie,
    "TURNO": turno,
    "SUELO": suelo,
    "PENDIENTE PROMEDIO FINCA": pendiente,
    "TOTAL DE ARBOLES": total_arboles,
    "DIAMETRO": diametro,
    "T PROGRAMADO": tiempo_programado,
    "HORAS DE OTRAS PARADA": horas_paradas,
    "clima_temp_promedio_dia_c": temp_prom,
    "clima_temp_min_dia_c": temp_min,
    "clima_temp_max_dia_c": temp_max,
    "clima_precipitacion_dia_mm": precipitacion,
    "clima_viento_promedio_dia_kmh": viento,
}

col1, col2 = st.columns([1.1, 1])

with col1:
    if modo == "Ranking de equipos":
        st.subheader("Ranking de equipos para el escenario")
        filtro = None if marca_filtro == "TODAS" else marca_filtro
        ranking = ranking_equipos(escenario, modelo=modelo, metadata=metadata, filtro_marca=filtro)

        if ranking.empty:
            st.warning("No hay equipos disponibles para la marca seleccionada.")
        else:
            mejor = ranking.iloc[0]
            st.metric(
                "Mejor equipo estimado",
                mejor["EQUIPO"],
                f"{mejor['PRODUCTIVIDAD_ESTIMADA_M3_H']:.2f} M3/HORA",
            )
            st.dataframe(ranking, use_container_width=True, hide_index=True)
            st.bar_chart(ranking.set_index("EQUIPO")["PRODUCTIVIDAD_ESTIMADA_M3_H"])
    else:
        st.subheader("Predicción para un equipo")
        pred = predecir_productividad(escenario, modelo=modelo, metadata=metadata)
        st.metric(
            "Productividad estimada",
            f"{pred['productividad_m3_h']:.2f} M3/HORA",
            pred["clasificacion"],
        )
        st.write("Entrada usada por el modelo:")
        st.dataframe(pd.DataFrame([pred["entrada_modelo"]]), use_container_width=True, hide_index=True)

with col2:
    st.subheader("Resumen del escenario")
    resumen = pd.DataFrame([escenario]).T.reset_index()
    resumen.columns = ["Variable", "Valor"]
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader("Métricas del modelo")
    metricas = metadata.get("metricas", {})
    prueba = metricas.get("prueba", {})
    cv = metricas.get("validacion_cruzada", {})
    st.write(
        {
            "MAE prueba": round(prueba.get("MAE", 0), 3),
            "RMSE prueba": round(prueba.get("RMSE", 0), 3),
            "R2 prueba": round(prueba.get("R2", 0), 3),
            "R2 CV": round(cv.get("R2_promedio", 0), 3),
        }
    )

st.divider()
st.caption(
    "Variables solicitadas: equipo, marca, zona, contratista, especie, turno, suelo, pendiente, árboles, "
    "diámetro, tiempo programado, horas de paradas y clima. Categorías no vistas se tratan como OTROS."
)
