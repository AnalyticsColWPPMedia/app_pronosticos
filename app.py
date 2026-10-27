"""App ilustrativa: pronóstico de series de tiempo con modelos estadísticos y de Machine Learning.
Curso de Series de Tiempo — Especialización en Analítica de Datos.

Ejecutar en local:   streamlit run app.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import modelos as M

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Pronósticos de series de tiempo", layout="wide")

COLORES = {"Naive estacional": "#6b6b6b", "ETS (automático)": "#2a78d6", "ARIMA (automático)": "#1baf7a", "LightGBM": "#eb6834"}
SERIES = {
    "Pasajeros aéreos (mensual, 1949–1960)": "datos/pasajeros_aereos.csv",
    "Tasa de desempleo, Colombia (mensual, %)": "datos/desempleo_colombia.csv",
    "Demanda de energía del SIN (diaria, GWh)": "datos/demanda_energia_diaria.csv",
}


# ----------------------------------------------------------------------------
# Carga de datos (con caché: solo se lee una vez)
# ----------------------------------------------------------------------------
@st.cache_data
def leer_csv(origen, col_fecha="fecha", col_valor="valor"):
    if isinstance(origen, str):                     # rutas relativas a la carpeta de la app
        origen = Path(__file__).parent / origen
    datos = pd.read_csv(origen)
    serie = pd.Series(pd.to_numeric(datos[col_valor], errors="coerce").values,
                      index=pd.to_datetime(datos[col_fecha]), name="valor").sort_index().dropna()
    serie = serie[~serie.index.duplicated()]
    freq = M.inferir_frecuencia(serie.index)
    serie = serie.asfreq(freq).interpolate()
    return serie, freq


@st.cache_data(show_spinner=False)
def calcular_pronostico(serie, freq, modelos, h, nivel):
    return M.pronosticar(serie, freq, list(modelos), h, nivel)


@st.cache_data(show_spinner=False)
def calcular_comparacion(serie, freq, modelos, h, n_ventanas):
    return M.comparar(serie, freq, list(modelos), h, n_ventanas)


@st.cache_data(show_spinner=False)
def calcular_monitoreo(serie, freq, modelo, corte, h_max, ventana):
    return M.monitorear(serie, freq, modelo, corte, h_max, ventana)


def grafico_base(titulo):
    fig = go.Figure()
    fig.update_layout(title=titulo, height=440, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified", template="plotly_white")
    return fig


# ----------------------------------------------------------------------------
# Barra lateral: elegir los datos
# ----------------------------------------------------------------------------
st.sidebar.title("Datos")
origen = st.sidebar.radio("¿Qué serie quiere usar?", ["Una serie del curso", "Subir mi propio CSV"])
if origen == "Una serie del curso":
    nombre_serie = st.sidebar.selectbox("Serie", list(SERIES))
    serie, freq = leer_csv(SERIES[nombre_serie])
else:
    archivo = st.sidebar.file_uploader("Archivo CSV con una columna de fecha y una de valor", type="csv")
    if archivo is None:
        st.info("Suba un archivo CSV en la barra lateral, o elija una serie del curso.")
        st.stop()
    columnas = pd.read_csv(archivo, nrows=5).columns.tolist()
    archivo.seek(0)
    col_fecha = st.sidebar.selectbox("Columna de fecha", columnas, index=0)
    col_valor = st.sidebar.selectbox("Columna de valor", columnas, index=min(1, len(columnas) - 1))
    try:
        serie, freq = leer_csv(archivo, col_fecha, col_valor)
    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")
        st.stop()
    nombre_serie = archivo.name

m = M.temporada(freq)
nombres_freq = {"MS": "mensual", "D": "diaria", "QS": "trimestral", "YS": "anual", "h": "horaria"}
st.sidebar.markdown(f"**{len(serie):,} observaciones** · frecuencia {nombres_freq.get(freq, freq)} · ciclo estacional de {m}")
st.sidebar.caption(f"Desde {serie.index.min():%Y-%m-%d} hasta {serie.index.max():%Y-%m-%d}")
if len(serie) < 3 * m + 10:
    st.sidebar.warning("La serie es corta para su frecuencia: los modelos estacionales pueden fallar.")

with st.sidebar.expander("¿Qué hay detrás de esta app?"):
    st.markdown(
        "- **Interfaz:** Streamlit (todo en Python).\n"
        "- **Modelos:** `statsforecast` (Naive, ETS, ARIMA) y `mlforecast` + LightGBM, los mismos del curso.\n"
        "- **Código separado:** la lógica de modelos está en `modelos.py`; la interfaz, en `app.py`.\n"
        "- **Despliegue:** el código vive en GitHub y Streamlit Community Cloud lo publica. "
        "Cada cambio en el repositorio actualiza la app.")

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("Pronósticos de series de tiempo")
st.caption("App ilustrativa del curso de Series de Tiempo: los modelos de los notebooks, usados sin escribir código.")

pestana_pron, pestana_comp, pestana_mon = st.tabs(["1 · Pronosticar", "2 · Comparar modelos", "3 · Monitorear un modelo"])

# ----------------------------------------------------------------------------
# 1. Pronosticar
# ----------------------------------------------------------------------------
with pestana_pron:
    st.markdown("Elija uno o varios modelos y el horizonte. La app entrena con **toda** la serie y pronostica hacia adelante.")
    c1, c2, c3 = st.columns([3, 2, 1])
    modelos_sel = c1.multiselect("Modelos", M.MODELOS, default=["ETS (automático)", "ARIMA (automático)"], key="mod_pron")
    h_def = 2 * m if m in (4, 12) else 28 if m == 7 else 10
    h = c2.slider("Horizonte (periodos a pronosticar)", 1, max(h_def * 3, 12), h_def, key="h_pron")
    nivel = c3.selectbox("Intervalo", [80, 95], index=1, format_func=lambda x: f"{x} %")

    if not modelos_sel:
        st.warning("Elija al menos un modelo.")
    else:
        with st.spinner("Entrenando modelos…"):
            pron = calcular_pronostico(serie, freq, tuple(modelos_sel), h, nivel)

        historia = serie.iloc[-max(6 * h, 4 * m, 60):]
        fig = grafico_base(f"{nombre_serie}: pronóstico a {h} periodos")
        fig.add_trace(go.Scatter(x=historia.index, y=historia.values, name="Histórico", line=dict(color="#222222", width=1.6)))
        for n in modelos_sel:
            color = COLORES[n]
            if f"{n}-lo" in pron:
                fig.add_trace(go.Scatter(x=list(pron.index) + list(pron.index[::-1]),
                                         y=list(pron[f"{n}-hi"]) + list(pron[f"{n}-lo"][::-1]),
                                         fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                         name=f"{n}: intervalo {nivel} %", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=pron.index, y=pron[n], name=n, line=dict(color=color, width=2.2)))
        st.plotly_chart(fig, width="stretch")

        tabla = pron.copy()
        tabla.index = tabla.index.strftime("%Y-%m-%d")
        tabla.index.name = "fecha"
        with st.expander("Ver la tabla del pronóstico"):
            st.dataframe(tabla.round(2), width="stretch")
        st.download_button("Descargar pronóstico (CSV)", tabla.round(4).to_csv().encode("utf-8"),
                           file_name="pronostico.csv", mime="text/csv")
        if "LightGBM" in modelos_sel and "LightGBM-lo" not in pron:
            st.caption("LightGBM no muestra intervalo: la serie es demasiado corta para calcular intervalos conformales.")

# ----------------------------------------------------------------------------
# 2. Comparar modelos
# ----------------------------------------------------------------------------
with pestana_comp:
    st.markdown("**Validación temporal con ventana expansiva:** la app entrena hasta un corte, pronostica, mide el error y repite "
                "con varios cortes. Así se elige el modelo por su desempeño fuera de la muestra, no por su ajuste.")
    c1, c2, c3 = st.columns([3, 2, 2])
    modelos_comp = c1.multiselect("Modelos a comparar", M.MODELOS, default=M.MODELOS, key="mod_comp")
    h_comp = c2.slider("Horizonte de cada prueba", 1, max(2 * m, 12), m if m > 1 else 4, key="h_comp")
    n_ventanas = c3.slider("Número de cortes", 2, 6, 3)

    if len(serie) - n_ventanas * h_comp < 3 * m + 10:
        st.warning("Con estos valores queda muy poca historia para entrenar. Reduzca el horizonte o el número de cortes.")
    elif st.button("Comparar modelos", type="primary") and modelos_comp:
        with st.spinner("Entrenando y evaluando en cada corte… (puede tardar unos segundos)"):
            res = calcular_comparacion(serie, freq, tuple(modelos_comp), h_comp, n_ventanas)
        resumen = res.groupby("modelo")[["MAE", "MAPE (%)", "MASE"]].mean().sort_values("MASE")
        mejor = resumen.index[0]
        st.success(f"Mejor modelo en promedio: **{mejor}** (MASE {resumen.loc[mejor, 'MASE']:.2f}).")

        c1, c2 = st.columns([1, 1])
        c1.markdown("**Error promedio en los cortes**")
        c1.dataframe(resumen.round(2), width="stretch")
        c1.caption("MASE < 1: mejor que el naive estacional dentro de la muestra. Menor es mejor en las tres métricas.")
        fig = go.Figure()
        for n in resumen.index:
            d = res[res["modelo"] == n]
            fig.add_trace(go.Bar(x=d["corte"].dt.strftime("%Y-%m-%d"), y=d["MASE"], name=n, marker_color=COLORES[n]))
        fig.update_layout(title="MASE en cada corte", barmode="group", height=360, template="plotly_white",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        c2.plotly_chart(fig, width="stretch")
        st.caption("Si el mejor modelo cambia de un corte a otro, la diferencia entre modelos probablemente no es concluyente.")

# ----------------------------------------------------------------------------
# 3. Monitorear
# ----------------------------------------------------------------------------
with pestana_mon:
    st.markdown("Un modelo en producción **se degrada**: el mundo cambia y el pronóstico se aleja de la realidad. "
                "Aquí se simula un modelo puesto en producción en una fecha pasada, se compara su pronóstico con lo que "
                "realmente ocurrió y se marca cuándo el error supera un umbral: el momento de **reentrenar**.")
    # Valores iniciales que cuentan una buena historia en cada serie del curso (el usuario puede cambiarlos)
    DEFECTOS = {"datos/pasajeros_aereos.csv": ("1957-12-01", "ETS (automático)", 10.0),
                "datos/desempleo_colombia.csv": ("2018-12-01", "ETS (automático)", 15.0),
                "datos/demanda_energia_diaria.csv": ("2025-04-30", "LightGBM", 6.0)}
    corte_def, modelo_def, umbral_def = DEFECTOS.get(SERIES.get(nombre_serie), (None, "ETS (automático)", 8.0 if m == 12 else 5.0))
    c1, c2, c3 = st.columns([2, 3, 2])
    modelo_mon = c1.selectbox("Modelo en producción", M.MODELOS, index=M.MODELOS.index(modelo_def), key=f"mod_mon_{nombre_serie}")
    fechas_posibles = serie.index[int(len(serie) * 0.4): len(serie) - max(m, 6)]
    pos_def = int(len(fechas_posibles) * 0.6) if corte_def is None else int(fechas_posibles.get_indexer([pd.Timestamp(corte_def)], method="nearest")[0])
    corte = c2.select_slider("Fecha en que se puso en producción", options=list(fechas_posibles), key=f"corte_{nombre_serie}",
                             value=fechas_posibles[pos_def], format_func=lambda d: d.strftime("%Y-%m-%d"))
    umbral = c3.slider("Umbral de alerta (error móvil, %)", 1.0, 30.0, umbral_def, 0.5, key=f"umbral_{nombre_serie}")
    h_max = 36 if m in (4, 12) else 180 if m == 7 else 3 * max(m, 10)
    ventana = 3 if m in (4, 12) else 7 if m == 7 else max(2, m // 4)

    with st.spinner("Simulando el modelo en producción…"):
        mon = calcular_monitoreo(serie, freq, modelo_mon, corte, h_max, ventana)

    alerta = mon.index[mon["error móvil %"] > umbral]
    fig = grafico_base("Pronóstico del modelo en producción vs. lo que realmente pasó")
    previo = serie.loc[:corte].iloc[-max(3 * ventana, 2 * m, 30):]
    fig.add_trace(go.Scatter(x=previo.index, y=previo.values, name="Datos de entrenamiento", line=dict(color="#222222", width=1.4)))
    fig.add_trace(go.Scatter(x=mon.index, y=mon["real"], name="Real (llega después)", line=dict(color="#6b6b6b", width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=mon.index, y=mon["pronóstico"], name=f"Pronóstico ({modelo_mon})", line=dict(color=COLORES[modelo_mon], width=2.2)))
    fig.add_vline(x=corte, line_dash="dash", line_color="#6b6b6b")
    if len(alerta):
        fig.add_vline(x=alerta[0], line_dash="dash", line_color="#e34948")
    st.plotly_chart(fig, width="stretch")

    fig2 = grafico_base(f"Error porcentual móvil ({ventana} periodos)")
    fig2.update_layout(height=280)
    fig2.add_trace(go.Scatter(x=mon.index, y=mon["error móvil %"], name="Error móvil %", line=dict(color="#eb6834", width=2.2)))
    fig2.add_hline(y=umbral, line_dash="dash", line_color="#e34948", annotation_text="umbral", annotation_position="top left")
    st.plotly_chart(fig2, width="stretch")

    if len(alerta):
        fecha_alerta = alerta[0]
        periodos = mon.index.get_loc(fecha_alerta) + 1
        st.error(f"**Alerta:** el {fecha_alerta:%Y-%m-%d}, {periodos} {'periodo' if periodos == 1 else 'periodos'} después de ponerlo en producción, "
                 f"el error móvil superó el {umbral:.1f} %. Es momento de reentrenar.")
        if st.checkbox("Reentrenar el modelo en la fecha de la alerta"):
            nuevo = calcular_monitoreo(serie, freq, modelo_mon, fecha_alerta, h_max, ventana)
            comun = nuevo.index.intersection(mon.index)
            nuevo = nuevo.loc[comun]          # comparar en el mismo periodo que el modelo original
            e_viejo, e_nuevo = mon.loc[comun, "error %"].mean(), nuevo.loc[comun, "error %"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Error del modelo original (desde la alerta)", f"{e_viejo:.2f} %")
            c2.metric("Error del modelo reentrenado", f"{e_nuevo:.2f} %", delta=f"{e_nuevo - e_viejo:+.2f} pp", delta_color="inverse")
            fig3 = grafico_base("Modelo original vs. reentrenado")
            fig3.update_layout(height=360)
            fig3.add_trace(go.Scatter(x=mon.index, y=mon["real"], name="Real", line=dict(color="#6b6b6b", width=1.8, dash="dot")))
            fig3.add_trace(go.Scatter(x=mon.index, y=mon["pronóstico"], name="Original", line=dict(color=COLORES[modelo_mon], width=2)))
            fig3.add_trace(go.Scatter(x=nuevo.index, y=nuevo["pronóstico"], name="Reentrenado", line=dict(color="#6250d6", width=2.2)))
            st.plotly_chart(fig3, width="stretch")
    else:
        st.success(f"El error móvil se mantuvo por debajo del {umbral:.1f} % en todo el periodo: no hizo falta reentrenar.")
    st.caption("En una empresa este monitoreo corre solo (por ejemplo, cada noche) y envía la alerta por correo; "
               "el reentrenamiento también puede ser automático. Eso es parte de lo que se llama MLOps.")
