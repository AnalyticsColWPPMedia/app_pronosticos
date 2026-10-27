"""Lógica de modelos de la app: pronóstico, validación temporal y monitoreo.

Toda la parte de modelos está aquí, separada de la interfaz (app.py). Es una buena práctica:
la misma lógica podría usarse desde un script programado, una API o un notebook.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals

MODELOS = ["Naive estacional", "ETS (automático)", "ARIMA (automático)", "LightGBM"]
_ALIAS_SF = {"Naive estacional": "SeasonalNaive", "ETS (automático)": "AutoETS", "ARIMA (automático)": "AutoARIMA"}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def inferir_frecuencia(fechas: pd.DatetimeIndex) -> str:
    """Frecuencia de la serie (MS mensual, D diaria, W semanal, QS trimestral, YS anual, h horaria)."""
    freq = pd.infer_freq(fechas)
    if freq is None:
        dias = pd.Series(fechas).diff().dt.days.median()
        freq = "D" if dias <= 1.5 else "W" if dias <= 8 else "MS" if dias <= 32 else "QS" if dias <= 95 else "YS"
    if freq.startswith(("M", "BM")):
        return "MS"
    if freq.startswith("W"):
        return freq
    if freq.startswith(("Q", "BQ")):
        return "QS"
    if freq.startswith(("A", "Y", "BA", "BY")):
        return "YS"
    return freq


def temporada(freq: str) -> int:
    """Largo del ciclo estacional según la frecuencia."""
    if freq == "MS":
        return 12
    if freq == "QS":
        return 4
    if freq.startswith("W"):
        return 52
    if freq in ("D", "B"):
        return 7
    if freq.lower() in ("h", "H"):
        return 24
    return 1


def a_nixtla(serie: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": "serie", "ds": serie.index, "y": serie.values.astype(float)})


def _lightgbm(m: int, n: int, freq: str) -> MLForecast:
    """LightGBM con rezagos de un ciclo y diferencia estacional (para que los árboles puedan seguir la tendencia)."""
    rezagos = list(range(1, m + 1)) if m <= 12 else [1, 2, 3, m, 2 * m, 3 * m, 4 * m]
    transformaciones = [Differences([m])] if m > 1 and n > 4 * m else [Differences([1])]
    modelo = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                               random_state=1, verbose=-1)
    return MLForecast(models={"LightGBM": modelo}, freq=freq, lags=rezagos, target_transforms=transformaciones)


def _metricas(real, pred, escala):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    mae = np.mean(np.abs(real - pred))
    return {"MAE": mae, "MAPE (%)": 100 * np.mean(np.abs((real - pred) / real)), "MASE": mae / escala}


# ----------------------------------------------------------------------------
# 1. Pronosticar
# ----------------------------------------------------------------------------
def pronosticar(serie: pd.Series, freq: str, modelos: list, h: int, nivel: int) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por fecha con columnas <modelo>, <modelo>-lo, <modelo>-hi."""
    m = temporada(freq)
    df = a_nixtla(serie)
    salida = pd.DataFrame(index=pd.date_range(serie.index[-1], periods=h + 1, freq=freq)[1:])

    est = [n for n in modelos if n in _ALIAS_SF]
    if est:
        clases = {"Naive estacional": SeasonalNaive(season_length=m), "ETS (automático)": AutoETS(season_length=m),
                  "ARIMA (automático)": AutoARIMA(season_length=m)}
        sf = StatsForecast(models=[clases[n] for n in est], freq=freq)
        p = sf.forecast(df=df, h=h, level=[nivel]).set_index("ds")
        for n in est:
            a = _ALIAS_SF[n]
            salida[n] = p[a].values
            salida[f"{n}-lo"] = p[f"{a}-lo-{nivel}"].values
            salida[f"{n}-hi"] = p[f"{a}-hi-{nivel}"].values

    if "LightGBM" in modelos:
        mlf = _lightgbm(m, len(serie), freq)
        ventanas_int = 3 if len(serie) > 6 * h + 4 * m else 0
        if ventanas_int:
            mlf.fit(df, prediction_intervals=PredictionIntervals(n_windows=ventanas_int, h=h))
            p = mlf.predict(h, level=[nivel])
            salida["LightGBM-lo"] = p[f"LightGBM-lo-{nivel}"].values
            salida["LightGBM-hi"] = p[f"LightGBM-hi-{nivel}"].values
        else:
            mlf.fit(df)
            p = mlf.predict(h)
        salida["LightGBM"] = p["LightGBM"].values
    return salida


# ----------------------------------------------------------------------------
# 2. Comparar (validación temporal)
# ----------------------------------------------------------------------------
def comparar(serie: pd.Series, freq: str, modelos: list, h: int, n_ventanas: int) -> pd.DataFrame:
    """Validación con ventana expansiva: entrena hasta cada corte, pronostica h y mide el error."""
    m = temporada(freq)
    df = a_nixtla(serie)
    filas = []
    for k in range(n_ventanas, 0, -1):
        fin = len(serie) - k * h
        entreno, prueba = serie.iloc[:fin], serie.iloc[fin:fin + h]
        escala = np.mean(np.abs(entreno.values[m:] - entreno.values[:-m])) if len(entreno) > m else np.mean(np.abs(np.diff(entreno.values)))
        p = pronosticar(entreno, freq, modelos, h, 80)
        for n in modelos:
            filas.append({"corte": entreno.index[-1], "modelo": n, **_metricas(prueba.values, p[n].values[:len(prueba)], escala)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 3. Monitorear
# ----------------------------------------------------------------------------
def monitorear(serie: pd.Series, freq: str, modelo: str, corte: pd.Timestamp, h_max: int, ventana: int) -> pd.DataFrame:
    """Simula un modelo puesto en producción en la fecha 'corte': pronostica hacia adelante y compara
    con lo que realmente pasó. Devuelve real, pronóstico, error porcentual y error móvil."""
    entreno = serie.loc[:corte]
    futuro = serie.loc[corte:].iloc[1:h_max + 1]
    p = pronosticar(entreno, freq, [modelo], len(futuro), 80)
    tabla = pd.DataFrame({"real": futuro.values, "pronóstico": p[modelo].values[:len(futuro)]}, index=futuro.index)
    tabla["error %"] = 100 * (tabla["real"] - tabla["pronóstico"]).abs() / tabla["real"]
    tabla["error móvil %"] = tabla["error %"].rolling(ventana, min_periods=ventana).mean()
    return tabla
