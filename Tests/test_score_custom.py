import sys
import os
from sklearn.metrics import confusion_matrix

# Récupérer le chemin du dossier parent (dossier 'API')
chemin_api = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ajouter le chemin du dossier 'API' au chemin de recherche de Python
sys.path.append(chemin_api)

from fonction import score_custom

def test_score_custom():
    # Exemple de valeurs pour les prédictions et les vraies valeurs
    y_val = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    # Calculer le score attendu en utilisant la fonction score_custom
    expected_score = score_custom(y_val, y_pred)

    # Score attendu pour les valeurs fournies
    assert expected_score == -9000