import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from pandas.io.xml import preprocess_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import MyException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration pour la transformation des données.
    Définit le chemin où l'objet de prétraitement sera sauvegardé.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Classe responsable de la transformation des données.
    - Gère la standardisation des colonnes numériques.
    - Gère l'encodage et l'imputation des colonnes catégoriques.
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transform_obj(self):
        """
        Crée et retourne un objet `ColumnTransformer` pour la transformation des données.

        Returns:
            ColumnTransformer: Un objet de transformation combinant pipelines numériques et catégoriels.

        Raises:
            MyException: En cas d'erreur dans la construction du pipeline.
        """
        try:
            logging.info("Début de la construction du pipeline de transformation des données.")

            # Définition des colonnes numériques et catégoriques
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline pour les variables numériques
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # Remplace les valeurs manquantes par la médiane
                ("scaler", StandardScaler()),  # Normalise les valeurs
            ])
            logging.info("Pipeline numérique créé avec imputation et standardisation.")

            # Pipeline pour les variables catégoriques
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplace les valeurs manquantes par la valeur la plus fréquente
                ("encoder", OneHotEncoder(handle_unknown="ignore")),  # Encodage One-Hot avec gestion des valeurs inconnues
                ("scaler", StandardScaler(with_mean=False)),  # Standardisation (évite l'erreur sur les matrices clairsemées)
            ])
            logging.info("Pipeline catégoriel créé avec imputation, encodage One-Hot et standardisation.")

            # Création du `ColumnTransformer`
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns),
            ])
            logging.info("Objet de transformation `ColumnTransformer` construit avec succès.")

            return preprocessor

        except Exception as e:
            logging.error(f"Erreur lors de la création du pipeline de transformation : {e}")
            raise MyException(e, sys)


    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Initialise la transformation des données :
        - Charge les fichiers de données d'entraînement et de test.
        - Applique les transformations définies.
        - Sauvegarde l'objet de prétraitement.

        Args:
            train_path (str): Chemin du fichier CSV contenant les données d'entraînement.
            test_path (str): Chemin du fichier CSV contenant les données de test.

        Returns:
            tuple: (train_arr, test_arr, chemin de l'objet de prétraitement)

        Raises:
            MyException: En cas d'erreur dans le processus.
        """
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Le fichier d'entraînement {train_path} est introuvable.")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Le fichier de test {test_path} est introuvable.")

            logging.info(f"Chargement des données d'entraînement depuis {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Chargement des données de test depuis {test_path}")
            test_df = pd.read_csv(test_path)

            logging.info("Lecture des fichiers CSV réussie.")
            logging.info("Obtention de l'objet de prétraitement...")

            preprocess_obj = self.get_data_transform_obj()
            logging.info("Objet de prétraitement obtenu avec succès.")

            # Définition des colonnes
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Séparation des features et de la variable cible
            logging.info("Séparation des features et de la variable cible.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transformation des données
            logging.info("Application de la transformation aux données d'entraînement et de test.")
            input_feature_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocess_obj.transform(input_feature_test_df)

            # Concaténation des features transformés avec la variable cible
            logging.info("Concaténation des données transformées avec la variable cible.")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Sauvegarde de l'objet de prétraitement
            logging.info(f"Sauvegarde de l'objet de prétraitement dans {self.config.preprocessor_obj_file_path}.")
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocess_obj)

            logging.info("Transformation des données terminée avec succès.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Erreur lors de la transformation des données : {e}")
            raise MyException(e, sys)
