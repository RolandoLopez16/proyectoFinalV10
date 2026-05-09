# Simulador de Productividad Harvester

Aplicación Streamlit basada en el notebook `RegresionLinealHV_v9.ipynb` para estimar productividad `M3/HORA` y comparar equipos Harvester bajo las mismas condiciones operativas.

## Qué hace

- Entrena un pipeline reproducible de regresión lineal.
- Aplica la misma lógica de limpieza del notebook v9:
  - selección de variables de negocio,
  - eliminación de objetivo nulo,
  - eliminación de duplicados,
  - eliminación de productividad no positiva,
  - control de outliers de `M3/HORA` con IQR,
  - agrupación de categorías poco frecuentes,
  - winsorización de variables numéricas,
  - imputación, escalamiento y One-Hot Encoding.
- Guarda artefactos con `joblib` y `json`.
- Permite simular escenarios y comparar equipos.

## Variables de entrada

Variables categóricas:

- EQUIPO
- MARCA
- ZONA
- CONTRATISTA
- ESPECIE
- TURNO
- SUELO

Variables numéricas:

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

## Ejecutar localmente

```bash
pip install -r requirements.txt
python -m src.entrenar
streamlit run app/streamlit_app.py
```

Si no ejecutas `python -m src.entrenar`, la app entrena automáticamente en el primer arranque.

## Dataset

La app busca el dataset en:

1. `data/Dataset_HV_2026_v2.xlsx`
2. raíz del proyecto
3. `/mnt/data/`
4. `/content/`
5. GitHub RAW: repositorio `RegresionLinealHV`

## Despliegue en Render

El repositorio incluye `render.yaml` y `Procfile`.

En Render usa:

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

## Nota técnica

La regresión lineal puede generar valores negativos si el escenario está fuera del rango histórico. La app conserva la predicción bruta para auditoría, pero muestra como valor operativo mínimo `0 M3/HORA`, porque la productividad real no puede ser negativa.
