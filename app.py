
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

def calcular_operacao(direcao, preco_entrada, preco_referencia):
    """
    Calcula entrada, stop, alvo1, alvo2 e R/R com base na direção e no padrão detectado.
    preco_referencia: fundo entre topos (VENDA) ou topo entre fundos (COMPRA)
    """
    if direcao == "VENDA":
        stop = round(preco_entrada * 1.02, 2)
        alvo1 = round(preco_referencia, 2)
        alvo2 = round(preco_entrada - (stop - preco_entrada) * 2, 2)
        risco = stop - preco_entrada
        retorno = preco_entrada - alvo1
    else:  # COMPRA
        stop = round(preco_entrada * 0.98, 2)
        alvo1 = round(preco_referencia, 2)
        alvo2 = round(preco_entrada + (preco_entrada - stop) * 2, 2)
        risco = preco_entrada - stop
        retorno = alvo1 - preco_entrada

    rr = round(retorno / risco, 2) if risco > 0 else 0
    return stop, alvo1, alvo2, rr

@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()

    # Features para o modelo
    features = [dados.get(c, 0) for c in colunas]
    X = np.array(features).reshape(1, -1)
    prob = modelo.predict_proba(X)[0][1]
    previsao = int(modelo.predict(X)[0])
    confianca = round(prob * 100, 1)

    # Direção baseada no tipo de padrão
    tipo_num = int(dados.get("tipo_num", -1))
    tipo_nome_map = {
        0: "TOPO DUPLO", 1: "TOPO TRIPLO", 2: "OCO",
        3: "FUNDO DUPLO", 4: "FUNDO TRIPLO", 5: "OCO INVERTIDO"
    }
    direcao_map = {
        0: "VENDA", 1: "VENDA", 2: "VENDA",
        3: "COMPRA", 4: "COMPRA", 5: "COMPRA"
    }

    direcao = direcao_map.get(tipo_num, "INDEFINIDO")
    tipo_nome = tipo_nome_map.get(tipo_num, "SEM PADRÃO")

    # Calcular operação completa
    preco_entrada = float(dados.get("preco_entrada", 0))
    preco_referencia = float(dados.get("preco_referencia", 0))

    if preco_entrada > 0 and preco_referencia > 0 and direcao != "INDEFINIDO":
        stop, alvo1, alvo2, rr = calcular_operacao(direcao, preco_entrada, preco_referencia)
    else:
        stop = alvo1 = alvo2 = rr = None

    return jsonify({
        "confianca": confianca,
        "acertou": previsao,
        "direcao": direcao,
        "tipo_padrao": tipo_nome,
        "entrada": preco_entrada,
        "stop": stop,
        "alvo1": alvo1,
        "alvo2": alvo2,
        "rr": rr
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "modelo": "B3 ML v1"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
