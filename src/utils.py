import os
import sys
import dill  # Utilisation de dill au lieu de pickle
import pickle
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
