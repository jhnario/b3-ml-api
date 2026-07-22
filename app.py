
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("b3_modelo.pkl", "rb") as f:
    salvo = pickle.load(f)

modelo = salvo["modelo"]
colunas = salvo["colunas_features"]
tipo_map = salvo["tipo_map"]

@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()
    features = [dados.get(c, 0) for c in colunas]
    X = np.array(features).reshape(1, -1)
    prob = modelo.predict_proba(X)[0][1]
    previsao = int(modelo.predict(X)[0])
    return jsonify({
        "acertou": previsao,
        "confianca": round(prob * 100, 1)
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "modelo": "B3 ML v1"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
