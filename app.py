from flask import Flask, request, jsonify
import numpy as np
import mlflow.pyfunc

app = Flask(__name__)

# Charger le modèle enregistré avec MLflow
model = mlflow.pyfunc.load_model("model")

@app.route('/predict', methods=['POST'])
def predict(data):
    try:
        best_tresh = 0.17061900216905868
        predictions = model.predict_proba(data)[:, 1]
        predictions = np.where(yhat >= best_tresh, "Refusé", "Accepté")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)