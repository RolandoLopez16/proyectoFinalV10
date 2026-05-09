"""Funciones de inferencia para la aplicación de productividad Harvester."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS


def asegurar_artefactos(auto_entrenar: bool = True) -> None:
    """Verifica artefactos; si no existen, entrena automáticamente."""
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        return

    if not auto_entrenar:
        raise FileNotFoundError("No existen artefactos del modelo. Ejecute: python -m src.entrenar")

    from src.entrenar import entrenar_y_guardar

    entrenar_y_guardar()


def cargar_sistema(auto_entrenar: bool = True):
    """Carga el pipeline y metadata."""
    asegurar_artefactos(auto_entrenar=auto_entrenar)
    modelo = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


def normalizar_categoria(valor: object, opciones: List[str]) -> str:
    """Normaliza entrada categórica y envía categorías no vistas a OTROS."""
    valor_norm = str(valor).strip().upper() if valor is not None else "OTROS"
    return valor_norm if valor_norm in opciones else "OTROS"


def preparar_entrada(entrada: Dict[str, object], metadata: Dict[str, object]) -> pd.DataFrame:
    """Convierte un diccionario de formulario en DataFrame válido para el pipeline."""
    fila: Dict[str, object] = {}

    opciones = metadata["opciones_categoricas"]
    defaults = metadata["numeric_defaults"]
    limites = metadata["limites_winsor"]

    for col in VARIABLES_CATEGORICAS:
        fila[col] = normalizar_categoria(entrada.get(col), opciones[col])

    for col in VARIABLES_NUMERICAS:
        valor = entrada.get(col, defaults[col])
        try:
            valor = float(valor)
        except (TypeError, ValueError):
            valor = float(defaults[col])

        li, ls = limites[col]
        fila[col] = min(max(valor, float(li)), float(ls))

    return pd.DataFrame([fila])[FEATURES]


def clasificar_productividad(valor: float, metadata: Dict[str, object]) -> str:
    """Clasifica la productividad usando percentiles del histórico limpio."""
    baja = metadata["umbrales_productividad"]["baja"]
    media = metadata["umbrales_productividad"]["media"]
    if valor < baja:
        return "Baja"
    if valor < media:
        return "Media"
    return "Alta"


def predecir_productividad(entrada: Dict[str, object], modelo=None, metadata=None) -> Dict[str, object]:
    """Predice productividad M3/HORA para un escenario."""
    if modelo is None or metadata is None:
        modelo, metadata = cargar_sistema()

    X = preparar_entrada(entrada, metadata)
    pred_bruta = float(modelo.predict(X)[0])

    # Regla operativa: la productividad no puede ser negativa.
    # Se conserva la predicción bruta para auditoría y se muestra una predicción operativa no negativa.
    pred_operativa = max(0.0, pred_bruta)

    return {
        "productividad_m3_h": pred_operativa,
        "productividad_modelo_bruta": pred_bruta,
        "clasificacion": clasificar_productividad(pred_operativa, metadata),
        "entrada_modelo": X.iloc[0].to_dict(),
    }


def equipos_disponibles(metadata: Dict[str, object], filtro_marca: Optional[str] = None) -> List[str]:
    """Lista equipos. Si hay marca seleccionada, filtra por la marca modal histórica."""
    equipos = metadata["opciones_categoricas"].get("EQUIPO", [])
    equipos = [e for e in equipos if e != "OTROS"]
    equipo_marca = metadata.get("equipo_marca_moda", {})

    if filtro_marca and filtro_marca != "TODAS":
        filtro = str(filtro_marca).strip().upper()
        equipos = [e for e in equipos if equipo_marca.get(e, "").upper() == filtro]

    return sorted(equipos)


def ranking_equipos(
    escenario: Dict[str, object],
    modelo=None,
    metadata=None,
    filtro_marca: Optional[str] = None,
) -> pd.DataFrame:
    """Compara equipos manteniendo constantes las demás condiciones."""
    if modelo is None or metadata is None:
        modelo, metadata = cargar_sistema()

    resultados = []
    equipo_marca = metadata.get("equipo_marca_moda", {})
    equipos = equipos_disponibles(metadata, filtro_marca=filtro_marca)

    for equipo in equipos:
        fila = dict(escenario)
        fila["EQUIPO"] = equipo
        # Evita combinaciones imposibles equipo-marca.
        fila["MARCA"] = equipo_marca.get(equipo, fila.get("MARCA", "OTROS"))
        pred = predecir_productividad(fila, modelo=modelo, metadata=metadata)
        resultados.append(
            {
                "EQUIPO": equipo,
                "MARCA": fila["MARCA"],
                "PRODUCTIVIDAD_ESTIMADA_M3_H": pred["productividad_m3_h"],
                "PREDICCION_BRUTA_M3_H": pred["productividad_modelo_bruta"],
                "CLASIFICACION": pred["clasificacion"],
            }
        )

    return pd.DataFrame(resultados).sort_values(
        "PRODUCTIVIDAD_ESTIMADA_M3_H", ascending=False
    ).reset_index(drop=True)
