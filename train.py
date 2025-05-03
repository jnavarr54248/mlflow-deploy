import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback
import joblib

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "CI-CD-Lab2"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Artifact Location Existente: {experiment.artifact_location} ---")
            if experiment.artifact_location != artifact_location:
                print(f"--- WARNING: Artifact location actual no coincide con la esperada ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento existente ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se obtuvo experiment_id válido ---")
    sys.exit(1)

# Entrenamiento
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: Artifact URI: {actual_artifact_uri} ---")

        # Verificación de ruta incorrecta
        if "/home/manuelcastiblan/" in actual_artifact_uri:
            print(f"--- ERROR CRÍTICO: URI del artefacto contiene ruta indebida ---")

        mlflow.log_metric("mse", mse)

        # Guardar modelo local
        model_path_absolute = os.path.abspath("model.pkl")
        try:
            joblib.dump(model, model_path_absolute)
            print("--- Debug: Modelo guardado localmente ---")
        except Exception as dump_err:
            print(f"--- ERROR al guardar el modelo: {dump_err} ---")
            traceback.print_exc()
            sys.exit(1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"✅ Modelo loggeado en MLflow. MSE: {mse:.4f}")

        # Guardar run_id para validación posterior
        with open("last_run_id.txt", "w") as f:
            f.write(run_id)
        print(f"--- Debug: run_id guardado en 'last_run_id.txt' ---")

except Exception as e:
    print(f"\n--- ERROR durante ejecución ---")
    traceback.print_exc()
    print(f"CWD actual: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {experiment_id}")
    if run:
        print(f"Run Artifact URI: {run.info.artifact_uri}")
    else:
        print("El objeto Run no se creó.")
    sys.exit(1)
