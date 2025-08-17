import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.train_model import (
    load_data, preprocesar_datos, agrupar_subastas,
    construir_dataset_modelo, cargar_modelo
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def test_model_r2():
    # ===== 1. Cargar y procesar datos =====
    df = load_data("./data/ebay.csv")
    df_proc = preprocesar_datos(df)
    df_agg  = agrupar_subastas(df_proc)
    X, y, _ = construir_dataset_modelo(df_agg)

    # ===== 2. Separar datos =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== 3. Cargar modelos entrenados desde la carpeta externa =====
    carpeta_modelos = r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"
    rf_model = cargar_modelo("RF", carpeta=carpeta_modelos)
    gb_model = cargar_modelo("GB", carpeta=carpeta_modelos)
    lr_model = cargar_modelo("LR", carpeta=carpeta_modelos)

    # ===== 4. Evaluar con R² =====
    r2_rf = r2_score(y_test, rf_model.predict(X_test))
    r2_gb = r2_score(y_test, gb_model.predict(X_test))
    r2_lr = r2_score(y_test, lr_model.predict(X_test))

    print(f"✅ Random Forest R²: {r2_rf:.2f}")
    print(f"✅ Gradient Boosting R²: {r2_gb:.2f}")
    print(f"✅ Regresión Lineal R²: {r2_lr:.2f}")

