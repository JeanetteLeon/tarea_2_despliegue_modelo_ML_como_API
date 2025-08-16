from model.train_model import (
    load_data, preprocesar_datos, agrupar_subastas,
    construir_dataset_modelo, entrenar_modelos
)

# Pipeline de entrenamiento
df = load_data("./data/ebay.csv")
df_proc = preprocesar_datos(df)
df_agg  = agrupar_subastas(df_proc)
X, y, df_modelo = construir_dataset_modelo(df_agg)

modelos, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)


