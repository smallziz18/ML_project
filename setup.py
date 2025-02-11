from typing import List

from setuptools import setup, find_packages


from typing import List
from pathlib import Path


def get_requirements(file_path: str) -> List[str]:
    """
    Lit un fichier de dépendances et retourne la liste des dépendances.

    Args :
        file_path (str): Le chemin du fichier contenant les dépendances (ex: 'requirements.txt').

    Returns :
        List[str]: Une liste contenant chaque dépendance sous forme de chaîne de caractères.

    Raises :
        FileNotFoundError : Si le fichier spécifié n'existe pas.
        IOError : En cas de problème de lecture du fichier.

    Example :
        >>> get_requirements("requirements.txt")
        ['numpy==1.21.2', 'pandas>=1.3.0', 'scikit-learn']
    """
    try:
        file = Path(file_path)  # Utilisation de Path pour une meilleure compatibilité
        if not file.exists():
            raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable.")

        with file.open("r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

        # Suppression de l'entrée '-e .' si présente
        return [req for req in requirements if req != '-e .']

    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        raise
    except IOError as e:
        print(f"Erreur de lecture du fichier '{file_path}' : {e}")
        raise


setup(
    name='ml-project-generique',
    version='0.0.1',
    packages=find_packages(),
    author='smallziz',
    author_email='abdoulazizdiouf221@gmail.com',
    install_requires=get_requirements('requirements.txt'),
)