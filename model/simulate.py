# model/simulate.py
from __future__ import annotations
from typing import Dict, Tuple
import random
import numpy as np
import pandas as pd

# === Debe coincidir EXACTO con las columnas usadas al entrenar ===
features_finales = [
    "precio_inicial_log", "reputacion_prom",
    "n_postores_log", "n_pujas_log",
    "producto_palm", "producto_xbox", "producto_cartier",
    "tipo_subasta_5", "tipo_subasta_7",
]


def transformar_entrada(subasta_dict: Dict) -> pd.DataFrame:
    """
    Transforma el diccionario de entrada a un DataFrame con
    las MISMAS columnas que vio el modelo en entrenamiento.
    """
    row = {}

    # logs
    row["precio_inicial_log"] = np.log1p(subasta_dict["precio_inicial"])
    row["n_postores_log"]     = np.log1p(subasta_dict.get("n_postores", 1))
    row["n_pujas_log"]        = np.log1p(subasta_dict.get("n_pujas", 1))

    # reputación promedio
    row["reputacion_prom"] = float(subasta_dict["reputacion_prom"])

    # dummies de producto
    for p in ["palm", "xbox", "cartier"]:
        row[f"producto_{p}"] = 1 if subasta_dict.get("producto") == p else 0

    # dummies de tipo_subasta (modelo usa 5 y 7)
    for t in [5, 7]:
        row[f"tipo_subasta_{t}"] = 1 if subasta_dict.get("tipo_subasta") == t else 0

    X = pd.DataFrame([row])

    # Asegurar TODAS las columnas y el orden correcto
    for col in features_finales:
        if col not in X.columns:
            X[col] = 0
    X = X[features_finales]

    return X


def predecir_todos(modelos: Dict[str, object], subasta: Dict) -> Dict[str, float]:
    """
    Devuelve las predicciones crudas (no-negativas) de cada modelo.
    """
    Xrow = transformar_entrada(subasta)
    preds = {nombre: max(0.0, float(m.predict(Xrow)[0])) for nombre, m in modelos.items()}
    return preds


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
    Simula 'juegos' subastas de segundo precio usando las predicciones
    de los modelos como base de la puja de cada jugador.

    Retorna:
      - resultados: conteo de victorias por modelo {'RF': n, 'GB': n, 'LR': n}
      - detalle: info de la última simulación
    """
    rng = random.Random(seed)
    resultados = {"RF": 0, "GB": 0, "LR": 0}
    ultimo_detalle: Dict = {}

    # transformar una sola vez
    Xrow = transformar_entrada(subasta)

    # loop de simulaciones
    for _ in range(juegos):
        # predicciones base (cap a >= 0)
        pred_rf = max(0.0, float(modelos["RF"].predict(Xrow)[0]))
        pred_gb = max(0.0, float(modelos["GB"].predict(Xrow)[0]))
        pred_lr = max(0.0, float(modelos["LR"].predict(Xrow)[0]))

        # regla de oferta: pred + 10 + 0.1*reputacion_prom (con tope)
        def oferta(p: float) -> int:
            return min(int(round(p + 10 + subasta["reputacion_prom"] * 0.1)), tope)

        ofertas = {
            "RF": oferta(pred_rf),
            "GB": oferta(pred_gb),
            "LR": oferta(pred_lr),
        }

        # valoraciones privadas
        val_priv = {k: rng.randint(valor_min, valor_max) for k in ofertas}

        # oferta efectiva (cap por valoración)
        ofertas_ef = {k: min(ofertas[k], val_priv[k]) for k in ofertas}

        # subasta de segundo precio
        orden = sorted(ofertas_ef.items(), key=lambda kv: kv[1], reverse=True)
        ganador, top = orden[0]
        segundo = orden[1][1] if len(orden) > 1 else 0
        precio_pagado = segundo + incremento if len(orden) > 1 else top

        resultados[ganador] += 1
        ultimo_detalle = {
            "subasta_input": subasta,
            "predicciones": {"RF": pred_rf, "GB": pred_gb, "LR": pred_lr},
            "valoraciones_privadas": val_priv,
            "ofertas": ofertas,
            "ofertas_efectivas": ofertas_ef,
            "ganador": ganador,
            "puja_max": float(top),
            "precio_pagado": float(precio_pagado),
        }

    return resultados, ultimo_detalle
