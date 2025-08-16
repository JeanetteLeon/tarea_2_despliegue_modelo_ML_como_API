import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.train_model import (
    load_data, preprocesar_datos, agrupar_subastas,
    construir_dataset_modelo, entrenar_modelos,
    guardar_modelos, cargar_modelo
)

# ===== Pipeline de entrenamiento =====
df = load_data("./data/ebay.csv")
df_proc = preprocesar_datos(df)
df_agg  = agrupar_subastas(df_proc)
X, y, df_modelo = construir_dataset_modelo(df_agg)

modelos, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)

# ===== Guardar modelos entrenados =====
guardar_modelos(modelos)

# (Opcional) cargar un modelo puntual para probar
rf_model = cargar_modelo("RF")
print(rf_model.predict(X_test[:5]))


