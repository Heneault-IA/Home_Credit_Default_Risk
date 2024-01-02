from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Charger le modèle enregistré avec MLflow
model_uri = f"runs:/0fdae83ac044403e9015817b7cfc3fc6/Projet7/Models/LGBM/Optimised"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        predictions = model.predict(data['input'])
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)