import os
import sys
import traceback
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# --- Config MLflow ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = tracking_uri

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "CI-CD-Obesity"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location)
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

# --- Cargar datos de obesidad ---
csv_path = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(csv_path)

# --- Preparación de datos ---
target_column = "NObeyesdad"  # Cambia esto si tu variable objetivo tiene otro nombre
X = df.drop(columns=[target_column])
y = df[target_column]

# Codificar variables categóricas
X = pd.get_dummies(X)
y = LabelEncoder().fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# --- Entrenamiento y registro ---
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    model_path = os.path.abspath("model.pkl")
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    with open("last_run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"✅ Modelo entrenado con éxito. Accuracy: {acc:.4f} | F1: {f1:.4f}")
