import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import MyException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration pour l'entraînement du modèle.

    Attributes:
        train_data_path (str): Chemin du fichier où sera sauvegardé le modèle entraîné.
    """
    train_data_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """
    Classe responsable de l'entraînement et de l'évaluation des modèles de régression.

    Methods:
        initiate_model_trainer(train_array, test_array, preprocessor_path):
            Entraîne différents modèles de Machine Learning et sauvegarde le meilleur modèle.
    """

    def __init__(self):
        """Initialise la configuration du ModelTrainer."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Entraîne plusieurs modèles de régression et sélectionne le meilleur.

        Args:
            train_array (numpy.ndarray): Tableau contenant les features et labels d'entraînement.
            test_array (numpy.ndarray): Tableau contenant les features et labels de test.
            preprocessor_path (str): Chemin du fichier contenant l'objet de préprocessing.

        Returns:
            float: Score R² du meilleur modèle sur les données de test.

        Raises:
            MyException: Si aucun modèle performant n'est trouvé.
        """
        try:
            logging.info('Initiating model trainer')

            # Séparation des features (X) et des labels (y) pour l'entraînement et le test
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Liste des modèles à entraîner
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Évaluation des modèles
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test, models=models)

            # Sélection du meilleur modèle
            best_model_score = max(model_report.values())  # Meilleur score R²
            best_model_name = max(model_report, key=model_report.get)  # Meilleur modèle
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise MyException("No best model found")  # Si le score est faible, on lève une exception

            logging.info(f"Best found model: {best_model_name} with score {best_model_score}")

            # Sauvegarde du meilleur modèle entraîné
            save_object(
                file_path=self.model_trainer_config.train_data_path,
                obj=best_model
            )

            # Prédiction avec le meilleur modèle
            predicted = best_model.predict(X_test)

            # Calcul du coefficient de détermination (R²) pour mesurer la performance du modèle
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except MyException as e:
            logging.error(f"Error in model training: {str(e)}")
            raise e
