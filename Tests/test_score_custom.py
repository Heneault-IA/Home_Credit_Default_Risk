from notebook import score_custom
from sklearn.metrics import confusion_matrix

def test_score_custom():
    # Exemple de valeurs pour les prédictions et les vraies valeurs
    y_val = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    # Calculer le score attendu en utilisant la fonction score_custom
    expected_score = score_custom(y_val, y_pred)

    # Score attendu pour les valeurs fournies
    assert expected_score == -30000  # Ici, on vérifie si le score calculé est -30000

    # Ajoutez plus de tests ici avec différents scénarios et valeurs attendues
    # par exemple : assert score_custom(autres_valeurs_y_val, autres_valeurs_y_pred) == score_attendu
