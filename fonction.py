def score_custom(y_val, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    score = fn * 10000 + fp * 1000 - tn * 10000

    return score