import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main():
    # Chemin des fichiers d'entrée et sortie
    processed_dir = "data/processed_data"
    models_dir = "models"

    X_train_path = f"{processed_dir}/X_train_scaled.csv"
    y_train_path = f"{processed_dir}/y_train.csv"

    # Creation du dossier de sortie s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)

    # Chargement des fichiers d'entrée
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Utilisation du modèle Random Forest
    model = RandomForestRegressor(random_state=42)

    # Application du Grid Search 
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    # Entrainement du Grid Search
    grid.fit(X_train, y_train)

    # Sauvegarde des meilleurs paramètres dans un pkl
    best_params = grid.best_params_
    joblib.dump(best_params, f"{models_dir}/best_params.pkl")

    # Affichage des logs du traitement
    print("Lancement du Grid Search effectuée --> Meilleurs paramètres sauvegardés dans models/best_params.pkl")
    print("Meilleurs paramètres:", best_params)

# Lancement du traitement principal uniquement
if __name__ == "__main__":
    main()
