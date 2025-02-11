import logging
import os
import sys
from datetime import datetime

from src.exception import MyException

# 📌 Création du nom du fichier log en fonction de la date et de l'heure actuelles
# Format : jour_mois_année_heure_minute_seconde.log
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# 📌 Définition du chemin du dossier où seront stockés les logs
# Ce dossier sera créé automatiquement s'il n'existe pas
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)  # Crée le dossier "logs" s'il n'existe pas

# 📌 Définition du chemin complet du fichier log
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 📌 Configuration du système de logging
logging.basicConfig(
    filename=LOG_FILE_PATH,  # 📍 Emplacement du fichier log
    level=logging.INFO,  # 🔹 Niveau de log : INFO (peut être modifié en DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - Ligne %(lineno)d - %(name)s - %(levelname)s - %(message)s',
    # 📍 Format des logs :
    # - %(asctime)s : Date et heure du log
    # - %(lineno)d : Numéro de ligne où le log a été généré
    # - %(name)s : Nom du logger
    # - %(levelname)s : Niveau du log (INFO, DEBUG, etc.)
    # - %(message)s : Message du log
)


