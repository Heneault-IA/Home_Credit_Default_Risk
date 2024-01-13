from ipaddress import collapse_addresses
from operator import index
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import mlflow.pyfunc

app = Flask(__name__)

# Charger le modèle enregistré avec MLflow
model = mlflow.pyfunc.load_model("model/model.plk")

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        best_tresh = 0.54

        file = request.files['file']
        if file:
            # Lire le fichier CSV
            df = pd.read_csv(file)

            # Faire des prédictions avec le modèle
            predictions = model.model.predict_proba(data)[:, 1]
            results = np.where(predictions >= best_tresh, "Refusé", "Accepté")

            # Ajouter les prédictions au DataFrame
            results = pd.DataFrame(results, index=df.index, columns=["Résultat"])

            # Convertir le DataFrame en HTML pour l'affichage
            result_html = results.to_html(classes='table table-striped', index=False)

            return render_template('result.html', tables=[result_html])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)