import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import MyException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    Configuration pour l'ingestion des données.
    Définit les chemins des fichiers de sortie pour les jeux de données.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    """
    Classe responsable de l'ingestion des données :
    - Chargement des données depuis un fichier CSV
    - Sauvegarde des données brutes
    - Division en train/test et sauvegarde
    """

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Exécute le processus d'ingestion des données.

        Returns:
            Tuple[str, str, str]: Chemins des fichiers train, test et raw.
        """
        logging.info('Début de l’ingestion des données')

        try:
            # Charger les données
            df = pd.read_csv('../../notebook/data/stud.csv')
            logging.info(f'Données chargées avec succès. Nombre d\'échantillons : {df.shape[0]}, Nombre de colonnes : {df.shape[1]}')

            # Création du dossier 'artifacts' si inexistant
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Sauvegarde des données brutes
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info(f'Données brutes sauvegardées sous {self.config.raw_data_path}')

            # Division en ensembles d'entraînement et de test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Sauvegarde des jeux de données
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info(f'Données d\'entraînement sauvegardées sous {self.config.train_data_path}')
            logging.info(f'Données de test sauvegardées sous {self.config.test_data_path}')
            logging.info('Ingestion des données terminée avec succès')

            return self.config.train_data_path, self.config.test_data_path, self.config.raw_data_path

        except Exception as e:
            logging.error(f"Erreur lors de l'ingestion des données : {str(e)}")
            raise MyException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train,test,raw = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train,test)