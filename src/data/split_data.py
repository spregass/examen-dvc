import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # Chemin des fichiers
    raw_path = "data/raw_data/raw.csv"
    processed_dir = "data/processed_data"

    # Créer le dossier de sortie si il n'existe pas
    os.makedirs(processed_dir, exist_ok=True)

    # Chargement du fichier d'entrée
    df = pd.read_csv(raw_path)

    # Split des datas en cible et entrée
    X = df.iloc[:, :-1]   # data entrée
    y = df.iloc[:, -1]    # data cible

    # Split des datas en données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegarde dans fichier de sortie
    X_train.to_csv(f"{processed_dir}/X_train.csv", index=False) # Fichier X_train
    X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)   # Fichier X_test
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False) # Fichier y_train
    y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)   # Fichier y_test

    # Affichage log du traitement
    print("Split data réalisé --> Fichiers sauvegardés dans data/processed_data/")

# Lancement de la fonction principale uniquement
if __name__ == "__main__":
    main()
