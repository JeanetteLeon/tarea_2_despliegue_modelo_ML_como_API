import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os



# =========================
# 1. Cargar datos
# =========================
def load_data(path="./data/ebay.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={
        df.columns[0]: 'id_subasta',
        df.columns[1]: 'monto_puja',
        df.columns[2]: 'tiempo_puja',
        df.columns[3]: 'user_postor',
        df.columns[4]: 'reputacion_postor',
        df.columns[5]: 'precio_inicial',
        df.columns[6]: 'precio_final',
        df.columns[7]: 'producto',
        df.columns[8]: 'tipo_subasta'
    })
    return df

# =========================
# 2. Preprocesar datos
# =========================
def preprocesar_datos(df):
    df = df.copy()
    df['id_subasta'] = df['id_subasta'].astype('category')
    df['reputacion_postor'] = df['reputacion_postor'].clip(lower=0, upper=100)
    df['monto_puja'] = df['monto_puja'].clip(upper=1000)
    df['monto_puja_log'] = np.log1p(df['monto_puja'])
    df['precio_inicial'] = df['precio_inicial'].clip(upper=1000)
    df['precio_inicial_log'] = np.log1p(df['precio_inicial'])
    return df

# =========================
# 3. Agrupar subastas
# =========================
def agrupar_subastas(df):
    df_agg = (df.groupby('id_subasta')
                .agg(precio_inicial=('precio_inicial', 'first'),
                     precio_final=('precio_final', 'max'),
                     reputacion_prom=('reputacion_postor', 'mean'),
                     producto=('producto', 'first'),
                     tipo_subasta=('tipo_subasta', 'first'))
                .reset_index())

    df_postores_pujas = (df.groupby('id_subasta')
                           .agg(n_postores=('user_postor', 'nunique'),
                                n_pujas=('user_postor', 'count'))
                           .reset_index())

    df_agg = df_agg.merge(df_postores_pujas, on='id_subasta', how='left')
    return df_agg

# =========================
# 4. Construir dataset modelo
# =========================
def construir_dataset_modelo(df_agg):
    cols_modelo_base = [
        'precio_inicial', 'reputacion_prom',
        'n_postores', 'n_pujas',
        'producto', 'tipo_subasta',
        'precio_final'
    ]
    df_modelo = df_agg[cols_modelo_base].copy()

    num_cols = ['precio_inicial','reputacion_prom','n_postores','n_pujas','precio_final']
    for c in num_cols:
        df_modelo[c] = pd.to_numeric(df_modelo[c], errors='coerce')
    df_modelo = df_modelo.dropna(subset=num_cols)

    df_modelo['producto'] = df_modelo['producto'].astype(str)
    df_modelo['tipo_subasta'] = df_modelo['tipo_subasta'].astype(str)

    items_validos = {'cartier','palm','xbox'}
    df_modelo = df_modelo[df_modelo['producto'].isin(items_validos)]

    df_modelo['precio_inicial_log'] = np.log1p(df_modelo['precio_inicial'])
    df_modelo['n_postores_log'] = np.log1p(df_modelo['n_postores'])
    df_modelo['n_pujas_log'] = np.log1p(df_modelo['n_pujas'])

    for c in ['precio_inicial_log','reputacion_prom','n_postores_log','n_pujas_log']:
        p99 = df_modelo[c].quantile(0.99)
        df_modelo[c] = np.minimum(df_modelo[c], p99)

    df_modelo = pd.get_dummies(df_modelo,
                               columns=['producto','tipo_subasta'],
                               drop_first=False)

    features_finales = [
        'precio_inicial_log', 'reputacion_prom',
        'n_postores_log', 'n_pujas_log',
        'producto_palm', 'producto_xbox', 'producto_cartier',
        'tipo_subasta_5', 'tipo_subasta_7'
    ]

    for col in features_finales:
        if col not in df_modelo:
            df_modelo[col] = 0

    X = df_modelo[features_finales].copy()
    y = df_modelo['precio_final'].astype(float)

    print("✅ Dataset construido")
    print("Features finales:", features_finales)
    print("X.shape:", X.shape, "| y.shape:", y.shape)

    return X, y, df_modelo

# =========================
# 5. Entrenar modelos
# =========================
def entrenar_modelos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)
    lr_model = LinearRegression()

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    modelos = {"RF": rf_model, "GB": gb_model, "LR": lr_model}
    print("✅ Modelos entrenados")
    return modelos, X_train, X_test, y_train, y_test



# =========================
# 6. Guardar y cargar modelos
# =========================
import os
import joblib

def guardar_modelos(modelos, carpeta=r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"):
    """
    Guarda los modelos entrenados en archivos .joblib dentro de la carpeta indicada.
    """
    os.makedirs(carpeta, exist_ok=True)
    for nombre, modelo in modelos.items():
        ruta = os.path.join(carpeta, f"{nombre}_modelo.joblib")
        joblib.dump(modelo, ruta)
        print(f"✅ Modelo {nombre} guardado en {ruta}")


def cargar_modelo(nombre, carpeta=r"C:\Users\jeane_bkpplgv\OneDrive\Documents\GitHub\Tareas_desarrollo_proyectos\modelo"):
    """
    Carga un modelo previamente guardado.
    """
    ruta = os.path.join(carpeta, f"{nombre}_modelo.joblib")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"⚠️ No se encontró el modelo en {ruta}")
    modelo = joblib.load(ruta)
    print(f"✅ Modelo {nombre} cargado desde {ruta}")
    return modelo
