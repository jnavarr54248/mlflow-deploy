import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys
import os

# Parámetro de umbral
THRESHOLD = 5000.0  # Ajusta según necesidad

# --- Cargar el dataset ---
print("--- Debug: Cargando dataset load_diabetes ---")
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Leer run_id desde archivo ---
run_id_path = "last_run_id.txt"
if not os.path.exists(run_id_path):
    print(f"❌ ERROR: No se encontró '{run_id_path}'. Asegúrate de ejecutar 'make train' primero.")
    sys.exit(1)

with open(run_id_path, "r") as f:
    run_id = f.read().strip()
print(f"--- Debug: run_id obtenido: {run_id} ---")

# --- Cargar modelo desde MLflow ---
model_uri = f"runs:/{run_id}/model"
print(f"--- Debug: Cargando modelo desde URI: {model_uri} ---")

try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    print(f"❌ ERROR al cargar modelo desde MLflow: {e}")
    sys.exit(1)

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
except ValueError as pred_err:
    print(f"❌ ERROR durante la predicción: {pred_err}")
    print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)

mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# Validación final
if mse <= THRESHOLD:
    print("✅ El modelo cumple con el umbral de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO cumple con el umbral de calidad. Fallando pipeline.")
    sys.exit(1)
