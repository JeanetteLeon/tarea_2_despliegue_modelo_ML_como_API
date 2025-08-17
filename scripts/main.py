# scripts/main.py
from typing import Dict, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, confloat, conint
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.train_model import cargar_modelo
from model.simulate import transformar_entrada, simular_subasta, predecir_todos

# ---------- Config ----------
MODELS_DIR = os.getenv("MODELS_DIR", "model/artifacts")

# Cols en el mismo orden del entrenamiento
FEATURES_FINALES = [
    "precio_inicial_log", "reputacion_prom",
    "n_postores_log", "n_pujas_log",
    "producto_palm", "producto_xbox", "producto_cartier",
    "tipo_subasta_5", "tipo_subasta_7"
]

# ---------- App ----------
app = FastAPI(
    title="API Precio Final de Subasta",
    version="1.0.0",
    description="Servicio de predicción (/predict) y simulación (/simulate) con modelos RF/GB/LR."
)

# Cargar modelos una vez al iniciar
MODELOS: Dict[str, object] = {}

@app.on_event("startup")
def _startup():
    global MODELOS
    MODELOS = {
        "RF": cargar_modelo("RF", carpeta=MODELS_DIR),
        "GB": cargar_modelo("GB", carpeta=MODELS_DIR),
        "LR": cargar_modelo("LR", carpeta=MODELS_DIR),
    }

# ---------- Esquemas ----------
class PredictPayload(BaseModel):
    precio_inicial: confloat(gt=0) = Field(..., description="Precio inicial de la subasta")
    n_postores: conint(ge=1) = Field(..., description="Número de postores")
    n_pujas: conint(ge=1) = Field(..., description="Número total de pujas")
    reputacion_prom: confloat(ge=0, le=100) = Field(..., description="Reputación promedio de postores")
    producto: Literal["xbox", "palm", "cartier"] = Field(..., description="Producto")
    tipo_subasta: Literal[3, 5, 7] = Field(..., description="Tipo de subasta (3, 5 o 7)")

class PredictResponse(BaseModel):
    predicciones: Dict[str, float]

class SimulatePayload(PredictPayload):
    juegos: conint(ge=1, le=10000) = 100
    valor_min: conint(ge=1) = 50
    valor_max: conint(gt=1) = 90
    incremento: conint(ge=1) = 1
    tope: conint(ge=1) = 100
    seed: int | None = 123

class SimulateResponse(BaseModel):
    resultados: Dict[str, int]
    ganador_global: Literal["RF", "GB", "LR"]
    detalle_ultimo_juego: Dict

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "models_dir": MODELS_DIR, "models": list(MODELOS.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload):
    # Clip a >= 0 y devolver tres modelos
    preds = predecir_todos(MODELOS, payload.dict())
    # (opcional) redondeo suave
    preds = {k: float(round(v, 4)) for k, v in preds.items()}
    return {"predicciones": preds}

@app.post("/simulate", response_model=SimulateResponse)
def simulate(payload: SimulatePayload):
    resultados, detalle = simular_subasta(
        modelos=MODELOS,
        subasta=payload.dict(),
        juegos=payload.juegos,
        valor_min=payload.valor_min,
        valor_max=payload.valor_max,
        incremento=payload.incremento,
        tope=payload.tope,
        seed=payload.seed,
    )
    # ganador global (modelo con más victorias)
    ganador_global = max(resultados.items(), key=lambda kv: kv[1])[0]
    return {
        "resultados": resultados,
        "ganador_global": ganador_global,
        "detalle_ultimo_juego": detalle,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.main:app", host="127.0.0.1", port=8000, reload=True)
