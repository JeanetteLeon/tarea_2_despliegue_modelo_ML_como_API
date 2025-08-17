# model/simulate.py
# ----------------------------------------------------
# Utilidades para transformar entradas y simular subastas
# ----------------------------------------------------

from __future__ import annotations
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# === Debe coincidir con las columnas con las que entrenaste ===
FEATURES_FINALES = [
    "precio_inicial_log", "reputacion_prom",
    "n_postores_log", "n_pujas_log",
    "producto_palm", "producto_xbox", "producto_cartier",
    "tipo_subasta_5", "tipo_subasta_7",
]


def transformar_entrada(subasta_dict: Dict) -> pd.DataFrame:
    """
    Convierte el diccionario de la subasta a un DataFrame con el MISMO
    conjunto de columnas (y orden) que se usó en entrenamiento.

    Espera llaves en español (ejemplo):
    {
        "precio_inicial": 45,
        "n_postores": 5,
        "n_pujas": 10,
        "reputacion_prom": 50,
        "producto": "xbox",        # uno de: "xbox" | "palm" | "cartier"
        "tipo_subasta": 5          # uno de: 3 | 5 | 7
    }
    """
    row: Dict[str, float | int] = {}

    # Logs de variables numéricas base
    row["precio_inicial_log"] = float(np.log1p(subasta_dict["precio_inicial"]))
    row["n_postores_log"]     = float(np.log1p(subasta_dict.get("n_postores", 1)))
    row["n_pujas_log"]        = float(np.log1p(subasta_dict.get("n_pujas", 1)))

    # Reputación promedio
    row["reputacion_prom"] = float(subasta_dict["reputacion_prom"])

    # Dummies de producto (3 categorías)
    prod = str(subasta_dict.get("producto", "")).lower()
    for p in ["palm", "xbox", "cartier"]:
        row[f"producto_{p}"] = 1 if prod == p else 0

    # Dummies de tipo_subasta (en el entrenamiento se usaron 5 y 7; 3 actúa como baseline)
    t = int(subasta_dict.get("tipo_subasta", 3))
    for tt in [5, 7]:
        row[f"tipo_subasta_{tt}"] = 1 if t == tt else 0

    # Crear DataFrame y asegurar columnas completas y en orden
    X = pd.DataFrame([row])
    for col in FEATURES_FINALES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURES_FINALES].copy()
    return X


def simular_subasta(
    modelos: Dict[str, object],
    subasta: Dict,
    juegos: int = 100,
    valor_min: int = 50,
    valor_max: int = 90,
    incremento: int = 1,
    tope: int = 100,
    seed: int = 123,
) -> Tuple[Dict[str, int], Dict]:
    """
    Simula una subasta de segundo precio usando las predicciones de los modelos.

    modelos: diccionario con los modelos entrenados, p.ej. {"RF": rf, "GB": gb, "LR": lr}
    subasta: payload en español (ver docstring de transformar_entrada)
    Retorna:
      - resultados: cuántas veces ganó cada modelo en 'juegos' iteraciones
      - detalle: información de la última iteración (útil para depurar/mostrar)
    """
    rng = random.Random(seed)
    resultados = {"RF": 0, "GB": 0, "LR": 0}
    ultimo_detalle: Dict = {}

    # Transformar una sola vez y asegurar orden de columnas
    Xrow = transformar_entrada(subasta)

    for _ in range(juegos):
        # Predicciones base
        pred_rf = float(modelos["RF"].predict(Xrow)[0])
        pred_gb = float(modelos["GB"].predict(Xrow)[0])
        pred_lr = float(modelos["LR"].predict(Xrow)[0])

        # Regla de oferta: predicción + 10 + 0.1 * reputacion_prom (tope)
        def oferta(p: float) -> int:
            return min(round(p + 10 + subasta["reputacion_prom"] * 0.1), tope)

        ofertas = {
            "RF": oferta(pred_rf),
            "GB": oferta(pred_gb),
            "LR": oferta(pred_lr),
        }

        # Valoraciones privadas por jugador
        val_priv = {k: rng.randint(valor_min, valor_max) for k in ofertas}

        # Oferta efectiva: no puede superar la valoración privada
        ofertas_ef = {k: min(ofertas[k], val_priv[k]) for k in ofertas}

        # Subasta de segundo precio
        orden = sorted(ofertas_ef.items(), key=lambda kv: kv[1], reverse=True)
        ganador, top = orden[0]
        segundo = orden[1][1] if len(orden) > 1 else 0
        precio_pagado = segundo + incremento if len(orden) > 1 else top

        resultados[ganador] += 1
        ultimo_detalle = {
            "subasta_input": subasta,
            "predicciones": {"RF": pred_rf, "GB": pred_gb, "LR": pred_lr},
            "valoraciones_privadas": val_priv,
            "ofertas_efectivas": ofertas_ef,
            "ganador": ganador,
            "puja_max": float(top),
            "precio_pagado": float(precio_pagado),
        }

    return resultados, ultimo_detalle


__all__ = ["FEATURES_FINALES", "transformar_entrada", "simular_subasta"]
