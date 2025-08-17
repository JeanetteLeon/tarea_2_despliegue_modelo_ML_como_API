# scripts/main_simulate.py (opcional para probar por consola)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.simulate import simular_subasta
from model.train_model import cargar_modelo

modelos = {
    "RF": cargar_modelo("RF", carpeta=r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"),
    "GB": cargar_modelo("GB", carpeta=r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"),
    "LR": cargar_modelo("LR", carpeta=r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"),
}

subasta = {
    "precio_inicial": 45,
    "n_postores": 5,
    "n_pujas": 10,
    "reputacion_prom": 50,
    "producto": "xbox",
    "tipo_subasta": 5,
}

res, detalle = simular_subasta(modelos, subasta, juegos=100, seed=123)
print(res)
print(detalle)
