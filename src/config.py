from pathlib import Path

RANDOM_STATE = 42
MIN_REGISTROS_CATEGORIA = 15

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

DATASET_FILE = "Dataset_HV_2026_v2.xlsx"
GITHUB_DATA_URL = "https://raw.githubusercontent.com/RolandoLopez16/RegresionLinealHV/main/Dataset_HV_2026_v2.xlsx"

MODEL_PATH = ARTIFACTS_DIR / "modelo_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
IMPORTANCE_PATH = ARTIFACTS_DIR / "importancia_variables.csv"
METRICS_PATH = ARTIFACTS_DIR / "metricas_modelo.json"

VARIABLES_CATEGORICAS = [
    "EQUIPO",
    "MARCA",
    "ZONA",
    "CONTRATISTA",
    "ESPECIE",
    "TURNO",
    "SUELO",
]

VARIABLES_NUMERICAS = [
    "PENDIENTE PROMEDIO FINCA",
    "TOTAL DE ARBOLES",
    "DIAMETRO",
    "T PROGRAMADO",
    "HORAS DE OTRAS PARADA",
    "clima_temp_promedio_dia_c",
    "clima_temp_min_dia_c",
    "clima_temp_max_dia_c",
    "clima_precipitacion_dia_mm",
    "clima_viento_promedio_dia_kmh",
]

TARGET = "M3/HORA"
FEATURES = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS
