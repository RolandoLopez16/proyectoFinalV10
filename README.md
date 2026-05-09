# Simulador de Productividad Harvester

Aplicación Streamlit basada en `RegresionLinealHV_v9.ipynb` para estimar productividad en M3/HORA y comparar equipos Harvester bajo condiciones operativas controladas.

## Objetivo

Convertir el análisis del notebook en una aplicación tipo MLOps:

- `src/entrenar.py`: entrena el pipeline y guarda artefactos.
- `src/prediccion.py`: carga el modelo y expone funciones de inferencia.
- `app/streamlit_app.py`: interfaz web para ingresar escenarios y comparar equipos.
- `artifacts/`: carpeta donde se guardan modelo, metadata, métricas e importancia de variables.

## Variables usadas

Categóricas:

- EQUIPO
- MARCA
- ZONA
- CONTRATISTA
- ESPECIE
- TURNO
- SUELO

Numéricas:

- PENDIENTE PROMEDIO FINCA
- TOTAL DE ARBOLES
- DIAMETRO
- T PROGRAMADO
- HORAS DE OTRAS PARADA
- clima_temp_promedio_dia_c
- clima_temp_min_dia_c
- clima_temp_max_dia_c
- clima_precipitacion_dia_mm
- clima_viento_promedio_dia_kmh

Variable objetivo:

- M3/HORA

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Entrenar modelo

```bash
python -m src.entrenar
```

El script carga `Dataset_HV_2026_v2.xlsx` desde `data/`, desde la raíz o desde GitHub RAW.

## Ejecutar app

```bash
streamlit run app/streamlit_app.py
```

## Despliegue en Render

Usa el archivo `render.yaml` o configura manualmente:

Build command:

```bash
pip install -r requirements.txt && python -m src.entrenar
```

Start command:

```bash
streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

## Interpretación

La app permite dos usos:

1. **Predicción de un equipo:** estima la productividad de un equipo específico.
2. **Ranking de equipos:** cambia únicamente el equipo y mantiene constantes las demás condiciones del escenario.

El ranking no significa que un equipo sea siempre mejor. Significa que, bajo el escenario ingresado, el modelo estima mayor productividad para ese equipo.

## Nota metodológica

El modelo se basa en regresión lineal y relaciones históricas del dataset. Sus resultados son asociaciones, no causalidad directa. Debe usarse como herramienta de apoyo a la decisión operativa.
