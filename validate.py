import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Umbral mínimo aceptable de F1-score para aprobar el modelo
F1_THRESHOLD = 0.80

# --- Cargar dataset de obesidad ---
csv_path = "ObesityDataSet_raw_and_data_sinthetic.csv"
if not os.path.exists(csv_path):
    print(f"❌ ERROR: No se encontró el dataset en {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
target_column = "NObeyesdad"  # Cambiar si tu variable objetivo tiene otro nombre

# Preparación de datos
X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(X)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# --- Predicción y métricas ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
except ValueError as pred_err:
    print(f"❌ ERROR durante la predicción: {pred_err}")
    print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)

# Calcular métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"🔍 Accuracy: {acc:.4f}")
print(f"🔍 Precision: {prec:.4f}")
print(f"🔍 Recall: {rec:.4f}")
print(f"🔍 F1-score: {f1:.4f} (umbral mínimo: {F1_THRESHOLD})")

# Validación final
if f1 >= F1_THRESHOLD:
    print("✅ El modelo cumple con el umbral de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO cumple con el umbral de calidad. Fallando pipeline.")
    sys.exit(1)
