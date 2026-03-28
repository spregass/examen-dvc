import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    # Chemin des fichiers d'entrée et de sortie
    processed_dir = "data/processed_data"
    models_dir = "models"

    X_train_path = f"{processed_dir}/X_train_scaled.csv"
    y_train_path = f"{processed_dir}/y_train.csv"
    best_params_path = f"{models_dir}/best_params.pkl"

    # Creation du dossier de sortie s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)

    # Chargement des fichiers d'entrée d'entrainement (normalisé)
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Chargement du pkl des meilleurs paramètres sauvegardés
    best_params = joblib.load(best_params_path)

    # Entrainement du modèle Random Forest avec les meilleurs paramètres
    model = RandomForestRegressor(
        random_state=42,
        **best_params
    )

    model.fit(X_train, y_train)

    # Sauvegarde du modèle entrainé dans un pkl
    joblib.dump(model, f"{models_dir}/model.pkl")

    # Affichage log du traitement
    print("Entrainement du modèle effectué --> Sauvegarde du modèle dans models/model.pkl")

# Lancement du traitement principal uniquement
if __name__ == "__main__":
    main()
