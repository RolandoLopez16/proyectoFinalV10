"""Entrenamiento reproducible del modelo de productividad Harvester.

Este script toma la lógica validada en RegresionLinealHV_v9.ipynb y la convierte
en un proceso reproducible: carga datos, limpia, preprocesa, entrena y guarda
artefactos para que la app Streamlit solo haga inferencia.
"""

from __future__ import annotations

import json
import os
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
    RANDOM_STATE,
    TARGET,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
)


def cargar_dataset() -> pd.DataFrame:
    """Carga el Excel desde data/, raíz del proyecto o GitHub RAW."""
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


def limpiar_y_preparar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Tuple[float, float]]]:
    """Aplica la limpieza y transformaciones previas al entrenamiento."""
    columnas = FEATURES + [TARGET]
    faltantes = [col for col in columnas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {faltantes}")

    data = df[columnas].copy()

    for col in VARIABLES_CATEGORICAS:
        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NULL": np.nan})
        )

    for col in VARIABLES_NUMERICAS + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[TARGET]).drop_duplicates()
    data = data[data[TARGET] > 0].copy()

    mapa_categorias: Dict[str, List[str]] = {}
    for col in VARIABLES_CATEGORICAS:
        frecuencias = data[col].value_counts(dropna=True)
        validas = frecuencias[frecuencias >= MIN_REGISTROS_CATEGORIA].index.tolist()
        data[col] = data[col].where(data[col].isin(validas), "OTROS")
        mapa_categorias[col] = sorted([str(x) for x in validas] + ["OTROS"])

    limites_winsor: Dict[str, Tuple[float, float]] = {}
    for col in VARIABLES_NUMERICAS:
        li = float(data[col].quantile(0.01)) if data[col].notna().any() else 0.0
        ls = float(data[col].quantile(0.99)) if data[col].notna().any() else 0.0
        data[col] = data[col].clip(lower=li, upper=ls)
        limites_winsor[col] = (li, ls)

    return data, mapa_categorias, limites_winsor


def construir_pipeline() -> Pipeline:
    """Crea el pipeline de preprocesamiento y regresión lineal."""
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
        ]
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


def construir_metadata(
    data: pd.DataFrame,
    mapa_categorias: Dict[str, List[str]],
    limites_winsor: Dict[str, Tuple[float, float]],
    metricas: Dict[str, object],
) -> Dict[str, object]:
    """Guarda información necesaria para construir formularios y validar entradas."""
    numeric_defaults = {col: float(data[col].median()) for col in VARIABLES_NUMERICAS}
    numeric_min = {col: float(data[col].quantile(0.01)) for col in VARIABLES_NUMERICAS}
    numeric_max = {col: float(data[col].quantile(0.99)) for col in VARIABLES_NUMERICAS}

    equipo_marca = (
        data.groupby("EQUIPO")["MARCA"]
        .agg(_moda_segura)
        .to_dict()
    )

    q33 = float(data[TARGET].quantile(0.33))
    q66 = float(data[TARGET].quantile(0.66))

    clima_seco = data[data["clima_precipitacion_dia_mm"].fillna(0) == 0]
    clima_lluvioso = data[data["clima_precipitacion_dia_mm"].fillna(0) > 0]

    def defaults_clima(subset: pd.DataFrame) -> Dict[str, float]:
        if subset.empty:
            subset = data
        return {
            "clima_temp_promedio_dia_c": float(subset["clima_temp_promedio_dia_c"].median()),
            "clima_temp_min_dia_c": float(subset["clima_temp_min_dia_c"].median()),
            "clima_temp_max_dia_c": float(subset["clima_temp_max_dia_c"].median()),
            "clima_precipitacion_dia_mm": float(subset["clima_precipitacion_dia_mm"].median()),
            "clima_viento_promedio_dia_kmh": float(subset["clima_viento_promedio_dia_kmh"].median()),
        }

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
            "SECO": defaults_clima(clima_seco),
            "LLUVIOSO": defaults_clima(clima_lluvioso),
        },
        "metricas": metricas,
        "nota": "Modelo de regresión lineal para estimar productividad M3/HORA. Las predicciones son apoyo analítico, no causalidad directa.",
    }


def entrenar_y_guardar() -> Dict[str, object]:
    """Ejecuta entrenamiento completo y guarda modelo + metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    data, mapa_categorias, limites_winsor = limpiar_y_preparar(df)

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)

    metricas = {
        "entrenamiento": _evaluar(modelo, X_train, y_train),
        "prueba": _evaluar(modelo, X_test, y_test),
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
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
        "RMSE_promedio": float(-cv["test_RMSE"].mean()),
        "R2_promedio": float(cv["test_R2"].mean()),
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

    metadata = construir_metadata(data, mapa_categorias, limites_winsor, metricas)

    joblib.dump(modelo, MODEL_PATH)
    importancia.to_csv(IMPORTANCE_PATH, index=False)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print("Métricas de prueba:", metricas["prueba"])
    return metricas


if __name__ == "__main__":
    entrenar_y_guardar()
