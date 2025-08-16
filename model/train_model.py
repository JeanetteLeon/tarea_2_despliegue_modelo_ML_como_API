# train_model.py
import numpy as np
import pandas as pd

# ==============================
# 1) Carga datos
# ==============================

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



# ==============================
# 2) Preprocesamiento
# ==============================

import numpy as np
import pandas as pd

def preprocesar_datos(df):
    """
    Preprocesa los datos crudos:
    - Ajusta tipos y clipping
    - Aplica transformaciones log
    """
    # --- 1) Ajustes básicos ---
    df['id_subasta'] = df['id_subasta'].astype('category')
    df['reputacion_postor'] = df['reputacion_postor'].clip(lower=0, upper=100)
    df['monto_puja'] = df['monto_puja'].clip(upper=1000)
    df['precio_inicial'] = df['precio_inicial'].clip(upper=1000)

    # --- 2) Transformaciones log ---
    df['monto_puja_log'] = np.log1p(df['monto_puja'])
    df['precio_inicial_log'] = np.log1p(df['precio_inicial'])

    return df


def agrupar_subastas(df):
    """
    Agrega información a nivel de subasta:
    - Variables clave por subasta
    - Número de postores y pujas
    """

    # --- Agregar por subasta ---
    df_agg = (
        df.groupby('id_subasta')
          .agg(
              precio_inicial=('precio_inicial', 'first'),
              precio_final=('precio_final', 'max'),
              reputacion_prom=('reputacion_postor', 'mean'),
              producto=('producto', 'first'),
              tipo_subasta=('tipo_subasta', 'first')
          )
          .reset_index()
    )

    # --- Contar postores únicos y número de pujas ---
    df_postores_pujas = (
        df.groupby('id_subasta')
          .agg(
              n_postores=('user_postor', 'nunique'),
              n_pujas=('user_postor', 'count')
          )
          .reset_index()
    )

    # --- Merge ---
    df_agg = df_agg.merge(df_postores_pujas, on='id_subasta', how='left')

    return df_agg

def construir_dataset_modelo(df_agg):
    """
    Construye el dataset final para el modelo ML:
    - Limpieza y casteo de tipos
    - Transformaciones log
    - Capping (p99)
    - One-hot encoding
    - Define X e y listos para entrenar
    """
    # Copia
    df_modelo = df_agg.copy()

    # --- 1) Tipos y limpieza ---
    num_cols = ['precio_inicial','reputacion_prom','n_postores','n_pujas','precio_final']
    for c in num_cols:
        df_modelo[c] = pd.to_numeric(df_modelo[c], errors='coerce')

    df_modelo = df_modelo.dropna(subset=num_cols)

    # categóricas
    df_modelo['producto'] = df_modelo['producto'].astype(str)
    df_modelo['tipo_subasta'] = df_modelo['tipo_subasta'].astype(str)

    # filtrar productos válidos
    items_validos = {'cartier','palm','xbox'}
    df_modelo = df_modelo[df_modelo['producto'].isin(items_validos)]

    # --- 2) Logs ---
    df_modelo['precio_inicial_log'] = np.log1p(df_modelo['precio_inicial'])
    df_modelo['n_postores_log']     = np.log1p(df_modelo['n_postores'])
    df_modelo['n_pujas_log']        = np.log1p(df_modelo['n_pujas'])

    # --- 3) Capping ---
    for c in ['precio_inicial_log','reputacion_prom','n_postores_log','n_pujas_log']:
        p99 = df_modelo[c].quantile(0.99)
        df_modelo[c] = np.minimum(df_modelo[c], p99)

    # --- 4) One-hot encoding ---
    df_modelo = pd.get_dummies(df_modelo,
                               columns=['producto','tipo_subasta'],
                               drop_first=False)

    # --- 5) Features finales ---
    features_finales = [
        'precio_inicial_log', 'reputacion_prom',
        'n_postores_log', 'n_pujas_log',
        'producto_palm', 'producto_xbox', 'producto_cartier',
        'tipo_subasta_5', 'tipo_subasta_7'
    ]

    # asegurar todas las columnas
    for col in features_finales:
        if col not in df_modelo:
            df_modelo[col] = 0

    # separar X e y
    X = df_modelo[features_finales].copy()
    y = df_modelo['precio_final'].astype(float)

    print("✅ Dataset construido")
    print("Features finales:", features_finales)
    print("X.shape:", X.shape, "| y.shape:", y.shape)

    return X, y, df_modelo






from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def entrenar_modelos(X, y, test_size=0.2, random_state=42):
    """
    Entrena Random Forest, Gradient Boosting y Regresión Lineal.
    
    Parámetros:
        X (DataFrame): Features
        y (Series): Target
        test_size (float): proporción de datos para test
        random_state (int): semilla reproducible
    
    Retorna:
        modelos (dict): {'RF': rf_model, 'GB': gb_model, 'LR': lr_model}
        X_train, X_test, y_train, y_test
    """
    # 1. Separar en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Inicializar modelos
    rf_model = RandomForestRegressor(random_state=random_state)
    gb_model = GradientBoostingRegressor(random_state=random_state)
    lr_model = LinearRegression()

    # 3. Entrenar
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    modelos = {"RF": rf_model, "GB": gb_model, "LR": lr_model}

    return modelos, X_train, X_test, y_train, y_test



