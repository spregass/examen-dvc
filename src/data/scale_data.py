import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

def main():
    # Chemin des fichiers d'entrée (et sortie)
    processed_dir = "data/processed_data"
    X_train_path = f"{processed_dir}/X_train.csv"
    X_test_path = f"{processed_dir}/X_test.csv"

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(processed_dir, exist_ok=True)

    # Chargement des fichiers d'entrée
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Retrait des colonnes non numeriques (a priori le timestamps)
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test.select_dtypes(include=["number"])

    # Normalisation des données d'entrée
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertion en Dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Sauvegarde dans les fichiers de sortie
    X_train_scaled.to_csv(f"{processed_dir}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{processed_dir}/X_test_scaled.csv", index=False)

    # Affichage log du traitement
    print("Normalisation data effectuée --> Fichiers sauvegardés dans data/processed_data/")

# Lancement de la fonction principale uniquement
if __name__ == "__main__":
    main()
