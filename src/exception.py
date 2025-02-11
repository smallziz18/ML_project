import sys


def error_message_detail(error, error_detail:sys):
    """
    Génère un message d'erreur détaillé incluant le fichier et la ligne où l'erreur s'est produite.

    :param error: L'exception capturée.
    :param error_detail: Le module `sys` utilisé pour récupérer les détails de l'erreur.
    :return: Un message d'erreur formaté.
    """
    _, _, exc_traceback = error_detail.exc_info()  # Récupère le traceback de l'exception

    file_name = exc_traceback.tb_frame.f_code.co_filename  # Nom du fichier où l'erreur s'est produite
    line_number = exc_traceback.tb_lineno  # Numéro de ligne où l'erreur est survenue

    error_message = "Une erreur s'est produite dans le script [{0}] ligne numéro [{1}] : [{2}]".format(
        file_name, line_number, str(error)
    )

    return error_message


class MyException(Exception):
    """
    Exception personnalisée qui inclut des détails supplémentaires sur l'erreur.

    :param error_message: Le message d'erreur initial.
    :param error_detail: Le module `sys` utilisé pour récupérer les détails de l'erreur.
    """

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
