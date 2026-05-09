"""Entrenamiento reproducible del modelo de productividad Harvester.

Este módulo replica el flujo validado en RegresionLinealHV_v9.ipynb:
1. carga del dataset,
2. selección de variables de negocio,
3. limpieza de objetivo, duplicados y valores inválidos,
4. control de outliers de productividad con IQR,
5. agrupación de categorías poco frecuentes,
6. winsorización de variables numéricas,
7. pipeline con imputación, escalamiento, One-Hot Encoding y regresión lineal,
8. evaluación train/test y validación cruzada,
9. guardado de artefactos para inferencia.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DATASET_FILE,
    FEATURES,
    GITHUB_DATA_URL,
    IMPORTANCE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MIN_REGISTROS_CATEGORIA,
    MODEL_PATH,
    N_SPLITS_CV,
    RANDOM_STATE,
    SAMPLE_PATH,
    TARGET,
    TEST_SIZE,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
)


def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset desde rutas locales o GitHub RAW."""
    rutas_locales = [
        DATA_DIR / DATASET_FILE,
        Path(DATASET_FILE),
        Path("/mnt/data") / DATASET_FILE,
        Path("/content") / DATASET_FILE,
    ]

    for ruta in rutas_locales:
        if ruta.exists():
            print(f"Dataset cargado desde archivo local: {ruta}")
            return pd.read_excel(ruta)

    print("Dataset local no encontrado. Cargando desde GitHub RAW...")
    return pd.read_excel(GITHUB_DATA_URL)


def _normalizar_categorica(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "<NA>": np.nan})
    )


def limpiar_y_preparar(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Tuple[float, float]], Dict[str, object]]:
    """Limpia y prepara datos antes del pipeline de sklearn."""
    columnas = FEATURES + [TARGET]
    faltantes = [col for col in columnas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {faltantes}")

    data = df[columnas].copy()
    resumen_limpieza: Dict[str, object] = {
        "filas_originales": int(len(data)),
        "columnas_modelado": int(len(columnas)),
    }

    for col in VARIABLES_CATEGORICAS:
        data[col] = _normalizar_categorica(data[col])

    for col in VARIABLES_NUMERICAS + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    filas_antes = len(data)
    data = data.dropna(subset=[TARGET]).copy()
    resumen_limpieza["filas_eliminadas_target_nulo"] = int(filas_antes - len(data))

    filas_antes = len(data)
    data = data.drop_duplicates().copy()
    resumen_limpieza["duplicados_eliminados"] = int(filas_antes - len(data))

    filas_antes = len(data)
    data = data[data[TARGET] > 0].copy()
    resumen_limpieza["filas_eliminadas_target_no_positivo"] = int(filas_antes - len(data))

    # Control de outliers de la variable objetivo, igual al notebook v9.
    q1 = float(data[TARGET].quantile(0.25))
    q3 = float(data[TARGET].quantile(0.75))
    iqr = q3 - q1
    limite_inf_target = q1 - 1.5 * iqr
    limite_sup_target = q3 + 1.5 * iqr

    filas_antes = len(data)
    data = data[(data[TARGET] >= limite_inf_target) & (data[TARGET] <= limite_sup_target)].copy()
    resumen_limpieza["limite_inferior_target_iqr"] = float(limite_inf_target)
    resumen_limpieza["limite_superior_target_iqr"] = float(limite_sup_target)
    resumen_limpieza["outliers_target_eliminados"] = int(filas_antes - len(data))

    mapa_categorias: Dict[str, List[str]] = {}
    for col in VARIABLES_CATEGORICAS:
        frecuencias = data[col].value_counts(dropna=True)
        validas = frecuencias[frecuencias >= MIN_REGISTROS_CATEGORIA].index.tolist()
        data[col] = data[col].where(data[col].isin(validas), "OTROS")
        opciones = sorted([str(x) for x in data[col].dropna().unique().tolist()])
        if "OTROS" not in opciones:
            opciones.append("OTROS")
        mapa_categorias[col] = opciones

    limites_winsor: Dict[str, Tuple[float, float]] = {}
    for col in VARIABLES_NUMERICAS:
        if data[col].notna().any():
            li = float(data[col].quantile(0.01))
            ls = float(data[col].quantile(0.99))
        else:
            li, ls = 0.0, 0.0
        data[col] = data[col].clip(lower=li, upper=ls)
        limites_winsor[col] = (li, ls)

    resumen_limpieza["filas_finales_modelado"] = int(len(data))
    return data, mapa_categorias, limites_winsor, resumen_limpieza


def construir_pipeline() -> Pipeline:
    """Construye el pipeline completo de sklearn."""
    pipeline_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, VARIABLES_NUMERICAS),
            ("cat", pipeline_categorico, VARIABLES_CATEGORICAS),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocesador),
            ("model", LinearRegression()),
        ]
    )


def _evaluar(modelo: Pipeline, X_data: pd.DataFrame, y_real: pd.Series) -> Dict[str, float]:
    pred = modelo.predict(X_data)
    return {
        "MAE": float(mean_absolute_error(y_real, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, pred))),
        "R2": float(r2_score(y_real, pred)),
    }


def _moda_segura(serie: pd.Series, defecto: str = "OTROS") -> str:
    moda = serie.dropna().mode()
    return str(moda.iloc[0]) if len(moda) else defecto


def _defaults_clima(data: pd.DataFrame, subset: pd.DataFrame) -> Dict[str, float]:
    if subset.empty:
        subset = data
    columnas_clima = [
        "clima_temp_promedio_dia_c",
        "clima_temp_min_dia_c",
        "clima_temp_max_dia_c",
        "clima_precipitacion_dia_mm",
        "clima_viento_promedio_dia_kmh",
    ]
    return {col: float(subset[col].median()) for col in columnas_clima}


def construir_metadata(
    data: pd.DataFrame,
    mapa_categorias: Dict[str, List[str]],
    limites_winsor: Dict[str, Tuple[float, float]],
    metricas: Dict[str, object],
    resumen_limpieza: Dict[str, object],
) -> Dict[str, object]:
    """Construye metadata para formularios, validación e interpretación."""
    numeric_defaults = {col: float(data[col].median()) for col in VARIABLES_NUMERICAS}
    numeric_min = {col: float(data[col].quantile(0.01)) for col in VARIABLES_NUMERICAS}
    numeric_max = {col: float(data[col].quantile(0.99)) for col in VARIABLES_NUMERICAS}

    equipo_marca = data.groupby("EQUIPO")["MARCA"].agg(_moda_segura).to_dict()
    q33 = float(data[TARGET].quantile(0.33))
    q66 = float(data[TARGET].quantile(0.66))

    clima_seco = data[data["clima_precipitacion_dia_mm"].fillna(0) <= 0.1]
    clima_lluvioso = data[data["clima_precipitacion_dia_mm"].fillna(0) > 0.1]

    return {
        "features": FEATURES,
        "target": TARGET,
        "variables_categoricas": VARIABLES_CATEGORICAS,
        "variables_numericas": VARIABLES_NUMERICAS,
        "opciones_categoricas": mapa_categorias,
        "numeric_defaults": numeric_defaults,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "limites_winsor": {k: [float(v[0]), float(v[1])] for k, v in limites_winsor.items()},
        "equipo_marca_moda": {str(k): str(v) for k, v in equipo_marca.items()},
        "umbrales_productividad": {"baja": q33, "media": q66},
        "clima_defaults": {
            "SECO": _defaults_clima(data, clima_seco),
            "LLUVIOSO": _defaults_clima(data, clima_lluvioso),
        },
        "metricas": metricas,
        "resumen_limpieza": resumen_limpieza,
        "nota": (
            "Pipeline alineado con RegresionLinealHV_v9: limpieza de objetivo, IQR, "
            "agrupación de categorías, winsorización, imputación, escalamiento, One-Hot Encoding y regresión lineal."
        ),
    }


def entrenar_y_guardar() -> Dict[str, object]:
    """Entrena el modelo y guarda artefactos."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    data, mapa_categorias, limites_winsor, resumen_limpieza = limpiar_y_preparar(df)

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)

    metricas = {
        "entrenamiento": _evaluar(modelo, X_train, y_train),
        "prueba": _evaluar(modelo, X_test, y_test),
    }

    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_validate(
        modelo,
        X,
        y,
        cv=kf,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
        return_train_score=False,
    )
    metricas["validacion_cruzada"] = {
        "MAE_promedio": float(-cv["test_MAE"].mean()),
        "MAE_desviacion": float(cv["test_MAE"].std()),
        "RMSE_promedio": float(-cv["test_RMSE"].mean()),
        "RMSE_desviacion": float(cv["test_RMSE"].std()),
        "R2_promedio": float(cv["test_R2"].mean()),
        "R2_desviacion": float(cv["test_R2"].std()),
    }

    perm = permutation_importance(
        modelo,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    importancia = pd.DataFrame(
        {
            "variable": X_test.columns,
            "importancia_MAE": perm.importances_mean,
            "desviacion": perm.importances_std,
        }
    ).sort_values("importancia_MAE", ascending=False)

    metadata = construir_metadata(data, mapa_categorias, limites_winsor, metricas, resumen_limpieza)

    joblib.dump(modelo, MODEL_PATH)
    importancia.to_csv(IMPORTANCE_PATH, index=False)
    data.sample(min(len(data), 200), random_state=RANDOM_STATE).to_csv(SAMPLE_PATH, index=False)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print("Métricas de prueba:", metricas["prueba"])
    print("Validación cruzada:", metricas["validacion_cruzada"])
    return metricas


if __name__ == "__main__":
    entrenar_y_guardar()
