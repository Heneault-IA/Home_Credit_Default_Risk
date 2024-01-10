from flask import Flask, render_template
import numpy as np
import mlflow.pyfunc

app = Flask(__name__)

# Charger le modèle enregistré avec MLflow
model = mlflow.pyfunc.load_model("model")

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(data):
    try:
        best_tresh = 0.17061900216905868
        predictions = model.predict_proba(data)[:, 1]
        predictions = np.where(yhat >= best_tresh, "Refusé", "Accepté")
        return predictions
    except Exception as e:
        return print(f'error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)