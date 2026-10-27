# App de pronósticos de series de tiempo

App ilustrativa del curso de **Series de Tiempo** (Especialización en Analítica de Datos). Usa los mismos modelos de los notebooks, sin escribir código:

1. **Pronosticar:** Naive estacional, ETS, ARIMA (automáticos, `statsforecast`) y LightGBM (`mlforecast`), con intervalos de predicción.
2. **Comparar modelos:** validación temporal con ventana expansiva (MAE, MAPE, MASE).
3. **Monitorear un modelo:** simula un modelo puesto en producción en una fecha pasada, compara su pronóstico con lo que realmente pasó, lanza una alerta cuando el error supera un umbral y muestra qué gana el modelo al reentrenarse.

Se puede usar con las tres series del curso o subiendo un CSV propio (una columna de fecha y una de valor).

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La interfaz (Streamlit). |
| `modelos.py` | La lógica de los modelos: pronóstico, validación y monitoreo. Separada de la interfaz a propósito. |
| `requirements.txt` | Librerías y versiones exactas que necesita la app. Streamlit Cloud las instala solo. |
| `datos/` | Las tres series del curso (la de energía, desde 2023, para que la app responda rápido). |

## Publicarla en internet (Streamlit Community Cloud, gratis, sin usar git)

**1. Subir el proyecto a GitHub**
1. Entre a [github.com](https://github.com) → botón **New** (repositorio nuevo).
2. Nombre: por ejemplo `app-pronosticos`. Visibilidad: **Public**. Marque *Add a README file*. **Create repository**.
3. En el repositorio: **Add file → Upload files**. Arrastre **el contenido** de esta carpeta (`app.py`, `modelos.py`, `requirements.txt`, `README.md` y la carpeta `datos` completa). Al arrastrar la carpeta `datos`, GitHub conserva la subcarpeta.
4. Abajo, **Commit changes**. Verifique que en el repositorio se vea `datos/` con los tres CSV adentro.

**2. Publicar la app**
1. Entre a [share.streamlit.io](https://share.streamlit.io) → **Continue with GitHub** y autorice el acceso.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repository: `su-usuario/app-pronosticos` · Branch: `main` · Main file path: `app.py`.
4. (Opcional) App URL: elija un nombre corto, por ejemplo `pronosticos-curso`.
5. **Advanced settings → Python version: 3.13** (o 3.12). Guardar.
6. **Deploy**. La primera vez tarda unos minutos instalando librerías. Al terminar, la app queda en `https://<nombre>.streamlit.app`.

**A tener en cuenta**
- La **primera ejecución** de cada modelo es lenta (unos 20–40 s): `statsforecast` compila su código la primera vez. Después responde rápido. Conviene abrir la app y pasar por las tres pestañas **antes de la clase**.
- Si nadie la usa por un tiempo, la app "se duerme". Al abrirla aparece un botón para despertarla (*Yes, get this app back up!*); tarda un minuto.
- La comparación con la serie diaria de energía es la más pesada (≈ 30–60 s con 4 modelos y 3 cortes).
- Cualquier cambio que se suba al repositorio de GitHub actualiza la app automáticamente.

## Ejecutarla en el propio computador (opcional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Formato del CSV propio

```
fecha,valor
2020-01-01,105.3
2020-02-01,98.7
...
```

Los nombres de las columnas pueden ser otros: la app deja elegir cuál es la fecha y cuál el valor. Frecuencias admitidas: diaria, semanal, mensual, trimestral, anual y horaria. La serie debe tener al menos tres ciclos estacionales completos (por ejemplo, tres años si es mensual).
