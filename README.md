
# API de Predicción y Simulación de Subastas de ebay

Este proyecto implementa una **API con FastAPI** para predecir el precio final de una subasta y simular el resultado de múltiples subastas.  
La API está desplegada en **Render** y es accesible públicamente. 

Permite:

- Predecir el **precio final esperado de una subasta** con tres modelos de Machine Learning (Random Forest, Gradient Boosting y Regresión Lineal).
- Simular subastas de **segundo precio** (subastas tipo ebay, que fue el dataset que se usó) con múltiples repeticiones, evaluando qué modelo gana más veces.

La API está desplegada en **Render** y es accesible públicamente:

## URL de la API en Render
- [https://tarea-2-despliegue-modelo-ml-como-api-2.onrender.com](https://tarea-2-despliegue-modelo-ml-como-api-2.onrender.com)

- Documentación interactiva con Swagger:  
  [https://tarea-2-despliegue-modelo-ml-como-api-2.onrender.com/docs](https://tarea-2-despliegue-modelo-ml-como-api-2.onrender.com/docs)


## Endpoints

###  1. Health Check
- **Ruta:** `/health`  
- **Método:** `GET`  
- **Descripción:** Verifica que la API y los modelos estén cargados correctamente.  
- **Ejemplo de respuesta:**

```json
{
  "status": "ok",
  "models_dir": "model/artifacts",
  "models": ["RF", "GB", "LR"]
}
```

### 2. Predicción de precio final
- **Ruta:** `/predict`  
- **Método:** `POST`  
- **Descripción:** Devuelve el precio estimado de la subasta según 3 modelos (Random Forest, Gradient Boosting y Regresión Lineal).



#### JSON de entrada esperado:
```json
{
  "precio_inicial": float (>0),
  "n_postores": int (>=1),
  "n_pujas": int (>=1),
  "reputacion_prom": float (0–100),
  "producto": "xbox" | "palm" | "cartier",
  "tipo_subasta": 3 | 5 | 7
}
```

#### Ejemplo válido:
```json
{
  "precio_inicial": 120,
  "n_postores": 4,
  "n_pujas": 15,
  "reputacion_prom": 70,
  "producto": "palm",
  "tipo_subasta": 7
}
```

#### Ejemplo de respuesta:
```json
{
  "predicciones": {
    "RF": 134.52,
    "GB": 120.48,
    "LR": 95.22
  }
}
```

---

### 3. Simulación

- **Ruta:** `/simulate`  
- **Método:** `POST`  
- **Descripción:** Ejecuta varias simulaciones de subastas y devuelve cuántas ganó cada modelo.

#### JSON de entrada esperado:
Incluye los mismos campos que `/predict` **más** parámetros de simulación:

```json
{
  "precio_inicial": float (>0),
  "n_postores": int (>=1),
  "n_pujas": int (>=1),
  "reputacion_prom": float (0–100),
  "producto": "xbox" | "palm" | "cartier",
  "tipo_subasta": 3 | 5 | 7,
  "juegos": int (1–10000),
  "valor_min": int,
  "valor_max": int,
  "incremento": int (>=1),
  "tope": int (>=1),
  "seed": int | null
}
```

#### Ejemplo válido:
```json
{
  "precio_inicial": 100,
  "n_postores": 3,
  "n_pujas": 20,
  "reputacion_prom": 50,
  "producto": "xbox",
  "tipo_subasta": 3,
  "juegos": 100,
  "valor_min": 50,
  "valor_max": 90,
  "incremento": 1,
  "tope": 100,
  "seed": 123
}
```

#### Ejemplo de respuesta:
```json
{
  "resultados": {
    "RF": 55,
    "GB": 45,
    "LR": 0
  },
  "detalle_ultimo_juego": {
    "subasta_input": {...},
    "predicciones": {"RF": 134.5, "GB": 120.4, "LR": 95.2},
    "valoraciones_privadas": {"RF": 67, "GB": 71, "LR": 53},
    "ofertas": {"RF": 100, "GB": 100, "LR": 37},
    "ofertas_efectivas": {"RF": 67, "GB": 71, "LR": 37},
    "ganador": "GB",
    "puja_max": 71.0,
    "precio_pagado": 68.0
  }
}
```

---

## ⚠️ Manejo de Errores

Si se envían datos inválidos, la API devuelve mensajes claros con el error.

#### Ejemplo:
```json
{
  "detail": [
    {
      "loc": ["body", "precio_inicial"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    }
  ]
}
```
---

## Requisitos para ejecutar localmente

- Crear entorno virtual

``` bash
conda create -n subastas_api python=3.12
conda activate subastas_api
```
- Instalar dependencias

``` bash
pip install -r requirements.txt
``` 

- Ejecutar la API

``` bash
uvicorn scripts.main:app --reload

``` 
La API quedará en:
http://127.0.0.1:8000/docs

---

##  Cliente de prueba

En este repositorio hay un **`client.ipynb`** (Jupyter Notebook) con ejemplos de:
- 1 consulta a `/health`
- 2 consultas distintas a `/predict`
- 1 consulta a `/simulate`

Esto demuestra cómo enviar datos y cómo interpretar las respuestas.

---

## ✅ Checklist de entrega
- [x] API desplegada en Render, accesible públicamente.  
- [x] Modelos cargados correctamente.  
- [x] Endpoints `/health`, `/predict`, `/simulate`.  
- [x] Validación de entradas con **mensajes claros** de error.  
- [x] Cliente de prueba (`client.ipynb`).  
- [x] Este README con **ejemplos de uso**.  

---

 Con esto el profesor podrá:  
1. Entrar a tu URL pública.  
2. Ver qué estructura de JSON debe enviar.  
3. Probar directamente `/predict` y `/simulate` en Swagger UI (`/docs`).  



📌 Autor: [Jeanette León Vejar]
📅 Curso: Desarrollo de Proyectos y Productos de Datos – Tarea 2