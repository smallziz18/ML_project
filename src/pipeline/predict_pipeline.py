import sys
import os
import pandas as pd
from src.exception import MyException
from src.utils import load_object


class PredictPipeline:
    """
    Classe responsable de la prédiction à l'aide du modèle entraîné.

    Methods:
        predict(features): Charge le modèle et le préprocesseur, applique la transformation et effectue une prédiction.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Effectue une prédiction en utilisant le modèle sauvegardé.

        Args:
            features (pd.DataFrame): Données d'entrée sous forme de DataFrame.

        Returns:
            np.ndarray: Prédictions du modèle.

        Raises:
            MyException: Si une erreur survient lors du chargement du modèle ou de la prédiction.
        """
        try:
            # Définition des chemins des fichiers
            model_path = "src/components/artifacts/model.pkl"
            preprocessor_path = 'src/components/artifacts/preprocessor.pkl'


            # Chargement du modèle et du préprocesseur
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)


            # Transformation des features
            data_scaled = preprocessor.transform(features)

            # Prédiction
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise MyException(e, sys)


class MyData:
    """
    Classe pour structurer les données d'entrée avant la prédiction.

    Attributes:
        gender (str): Sexe de l'élève.
        race_ethnicity (str): Groupe ethnique de l'élève.
        parental_level_of_education (str): Niveau d'éducation des parents.
        lunch (str): Type de déjeuner (standard ou réduit).
        test_preparation_course (str): Participation à un cours de préparation.
        reading_score (int): Score de lecture.
        writing_score (int): Score d'écriture.

    Methods:
        get_data_as_data_frame(): Retourne les données sous forme d'un DataFrame Pandas.
    """

    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):
        """
        Initialise une instance de MyData avec les attributs de l'élève.

        Args:
            gender (str): Sexe de l'élève.
            race_ethnicity (str): Groupe ethnique de l'élève.
            parental_level_of_education (str): Niveau d'éducation des parents.
            lunch (str): Type de déjeuner (standard ou réduit).
            test_preparation_course (str): Participation à un cours de préparation.
            reading_score (int): Score de lecture.
            writing_score (int): Score d'écriture.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convertit les attributs en un DataFrame Pandas.

        Returns:
            pd.DataFrame: Données structurées sous forme de DataFrame.

        Raises:
            MyException: Si une erreur survient lors de la conversion des données.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise MyException(e, sys)
