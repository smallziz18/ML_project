import os
import sys
import dill  # Utilisation de dill au lieu de pickle
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import MyException
from src.logger import logging  # Importation du logger

def save_object(file_path: str, obj) -> None:
    """
    Sauvegarde un objet sérialisé dans un fichier.

    Args:
        file_path (str): Chemin où enregistrer l'objet.
        obj: L'objet à sauvegarder.

    Raises:
        MyException: En cas d'erreur lors de la sauvegarde.
    """
    try:
        if not file_path:
            raise ValueError("Le chemin du fichier ne peut pas être vide.")
        if obj is None:
            raise ValueError("L'objet à sauvegarder ne peut pas être None.")

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Utilisation de dill pour une meilleure compatibilité

        logging.info(f"Objet sauvegardé avec succès dans {file_path}")

    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de l'objet : {e}")
        raise MyException(e, sys)




from sklearn.metrics import r2_score
import sys
from src.exception import MyException

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Entraîne et évalue plusieurs modèles de Machine Learning en utilisant le coefficient de détermination R².

    Args:
        X_train (numpy.ndarray ou pd.DataFrame): Features du jeu d'entraînement.
        y_train (numpy.ndarray ou pd.Series): Labels cibles du jeu d'entraînement.
        X_test (numpy.ndarray ou pd.DataFrame): Features du jeu de test.
        y_test (numpy.ndarray ou pd.Series): Labels cibles du jeu de test.
        models (dict): Dictionnaire contenant les modèles à évaluer.
                       Clés = noms des modèles, Valeurs = instances des modèles.

    Returns:
        dict: Un dictionnaire où les clés sont les noms des modèles et les valeurs sont leurs scores R² sur les données de test.

    Raises:
        MyException: En cas d'erreur lors de l'entraînement ou de l'évaluation des modèles.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> models = {
        ...     "Linear Regression": LinearRegression(),
        ...     "Random Forest": RandomForestRegressor()
        ... }
        >>> report = evaluate_models(X_train, y_train, X_test, y_test, models)
        >>> print(report)
        {"Linear Regression": 0.85, "Random Forest": 0.92}
    """

    try:
        report = {}  # Dictionnaire pour stocker les scores R² des modèles

        for i in range(len(list(models))):  # Parcourt la liste des modèles
            model = list(models.values())[i]  # Récupère le modèle actuel

            model.fit(X_train, y_train)  # Entraîne le modèle sur les données d'entraînement

            y_train_pred = model.predict(X_train)  # Prédictions sur l'entraînement
            y_test_pred = model.predict(X_test)  # Prédictions sur le test

            train_model_score = r2_score(y_train, y_train_pred)  # Calcul du score R² sur train
            test_model_score = r2_score(y_test, y_test_pred)  # Calcul du score R² sur test

            report[list(models.keys())[i]] = test_model_score  # Stocke le score R² de test dans le dictionnaire

        return report  # Retourne le dictionnaire contenant les scores des modèles

    except Exception as e:
        raise MyException(e, sys)  # Capture et lève une exception personnalisée en cas d'erreur
