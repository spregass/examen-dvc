import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    # Chemin des fichiers d'entrée et de sortie
    processed_dir = "data/processed_data"
    models_dir = "models"
    metrics_dir = "metrics"
    data_dir = "data"

    X_test_path = f"{processed_dir}/X_test_scaled.csv"
    y_test_path = f"{processed_dir}/y_test.csv"
    model_path = f"{models_dir}/model.pkl"

    # Creation des dossiers de sortie s'ils n'existent pas
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Chargement des fichiers d'entrée
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Chargement du modèle pkl entrainé avec les meilleurs paramètres
    model = joblib.load(model_path)

    # Prédiction
    y_pred = model.predict(X_test)

    # Métriques  (mse, rmse et r2)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    # Sauvegarde des métriques dans scores.json
    with open(f"{metrics_dir}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    # Sauvegarde des prédictions dans un csv
    pred_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred
    })
    pred_df.to_csv(f"{data_dir}/predictions.csv", index=False)

    # Affichage des logs du traitement
    print("Evaluation du modèle effectué --> Métriques sauvegardés dans metrics/scores.json")
    print("                              --> Predictions sauvegardées to data/predictions.csv")

# Lancement du traitement principal uniquement
if __name__ == "__main__":
    main()
