
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

def detectar_topos_fundos(closes, janela=10):
    topos, fundos = [], []
    for i in range(janela, len(closes) - janela):
        if closes[i] > max(closes[i-janela:i]) and closes[i] > max(closes[i+1:i+janela+1]):
            topos.append({"idx": i, "preco": closes[i]})
        if closes[i] < min(closes[i-janela:i]) and closes[i] < min(closes[i+1:i+janela+1]):
            fundos.append({"idx": i, "preco": closes[i]})
    return topos, fundos

def detectar_padrao(topos, fundos, tolerancia=0.08):
    candidatos = []

    # Topo duplo
    for i in range(len(topos)-1):
        t1, t2 = topos[i], topos[i+1]
        if abs(t1["preco"] - t2["preco"]) / t1["preco"] <= tolerancia:
            f_entre = [f for f in fundos if t1["idx"] < f["idx"] < t2["idx"]]
            if f_entre:
                candidatos.append((t2["idx"], 0, "TOPO DUPLO", "VENDA", t2["preco"], f_entre[0]["preco"]))

    # Topo triplo
    for i in range(len(topos)-2):
        t1, t2, t3 = topos[i], topos[i+1], topos[i+2]
        if abs(t1["preco"]-t2["preco"])/t1["preco"] <= tolerancia and abs(t2["preco"]-t3["preco"])/t2["preco"] <= tolerancia:
            f1 = [f for f in fundos if t1["idx"] < f["idx"] < t2["idx"]]
            f2 = [f for f in fundos if t2["idx"] < f["idx"] < t3["idx"]]
            if f1 and f2:
                candidatos.append((t3["idx"], 1, "TOPO TRIPLO", "VENDA", t3["preco"], min(f1[0]["preco"], f2[0]["preco"])))

    # OCO
    for i in range(len(topos)-2):
        oe, cab, od = topos[i], topos[i+1], topos[i+2]
        if cab["preco"] > oe["preco"] and cab["preco"] > od["preco"]:
            if abs(oe["preco"]-od["preco"])/oe["preco"] <= 0.05:
                f1 = [f for f in fundos if oe["idx"] < f["idx"] < cab["idx"]]
                f2 = [f for f in fundos if cab["idx"] < f["idx"] < od["idx"]]
                if f1 and f2:
                    pescoco = (f1[0]["preco"] + f2[0]["preco"]) / 2
                    candidatos.append((od["idx"], 2, "OCO", "VENDA", od["preco"], pescoco))

    # Fundo duplo
    for i in range(len(fundos)-1):
        f1, f2 = fundos[i], fundos[i+1]
        if abs(f1["preco"] - f2["preco"]) / f1["preco"] <= tolerancia:
            t_entre = [t for t in topos if f1["idx"] < t["idx"] < f2["idx"]]
            if t_entre:
                candidatos.append((f2["idx"], 3, "FUNDO DUPLO", "COMPRA", f2["preco"], t_entre[0]["preco"]))

    # Fundo triplo
    for i in range(len(fundos)-2):
        f1, f2, f3 = fundos[i], fundos[i+1], fundos[i+2]
        if abs(f1["preco"]-f2["preco"])/f1["preco"] <= tolerancia and abs(f2["preco"]-f3["preco"])/f2["preco"] <= tolerancia:
            t1 = [t for t in topos if f1["idx"] < t["idx"] < f2["idx"]]
            t2 = [t for t in topos if f2["idx"] < t["idx"] < f3["idx"]]
            if t1 and t2:
                candidatos.append((f3["idx"], 4, "FUNDO TRIPLO", "COMPRA", f3["preco"], max(t1[0]["preco"], t2[0]["preco"])))

    # OCO invertido
    for i in range(len(fundos)-2):
        oe, cab, od = fundos[i], fundos[i+1], fundos[i+2]
        if cab["preco"] < oe["preco"] and cab["preco"] < od["preco"]:
            if abs(oe["preco"]-od["preco"])/oe["preco"] <= 0.05:
                t1 = [t for t in topos if oe["idx"] < t["idx"] < cab["idx"]]
                t2 = [t for t in topos if cab["idx"] < t["idx"] < od["idx"]]
                if t1 and t2:
                    pescoco = (t1[0]["preco"] + t2[0]["preco"]) / 2
                    candidatos.append((od["idx"], 5, "OCO INVERTIDO", "COMPRA", od["preco"], pescoco))

    if not candidatos:
        return -1, "SEM PADRAO", "INDEFINIDO", 0, 0

    # Retorna o padrao mais recente (maior idx)
    candidatos.sort(key=lambda x: x[0], reverse=True)
    _, tipo_num, tipo_nome, direcao, preco_entrada, preco_referencia = candidatos[0]
    return tipo_num, tipo_nome, direcao, preco_entrada, preco_referencia

def calcular_features(candles):
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    opens  = [c["open"]  for c in candles]
    vols   = [c["volume"] for c in candles]

    mm20  = np.mean(closes[-20:])  if len(closes) >= 20  else np.mean(closes)
    mm50  = np.mean(closes[-50:])  if len(closes) >= 50  else np.mean(closes)
    mm200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
    vol_med = np.mean(vols[-20:])  if len(vols)   >= 20  else np.mean(vols)

    close = closes[-1]
    preco_vs_mm20  = (close - mm20)  / mm20  * 100 if mm20  else 0
    preco_vs_mm50  = (close - mm50)  / mm50  * 100 if mm50  else 0
    preco_vs_mm200 = (close - mm200) / mm200 * 100 if mm200 else 0
    mm20_vs_mm50   = (mm20  - mm50)  / mm50  * 100 if mm50  else 0
    volume_vs_media = vols[-1] / vol_med if vol_med else 1

    amplitude = highs[-1] - lows[-1] if highs[-1] != lows[-1] else 0.0001
    corpo      = abs(close - opens[-1])
    sombra_sup = highs[-1] - max(close, opens[-1])
    sombra_inf = min(close, opens[-1]) - lows[-1]
    sombra_sup_pct = sombra_sup / amplitude * 100
    sombra_inf_pct = sombra_inf / amplitude * 100
    corpo_pct      = corpo      / amplitude * 100

    topo_fib  = max(highs[-60:]) if len(highs) >= 60 else max(highs)
    fundo_fib = min(lows[-60:])  if len(lows)  >= 60 else min(lows)
    diff = topo_fib - fundo_fib if topo_fib != fundo_fib else 0.0001
    dist_fib382 = abs(close - (topo_fib - diff * 0.382)) / close * 100
    dist_fib500 = abs(close - (topo_fib - diff * 0.500)) / close * 100
    dist_fib618 = abs(close - (topo_fib - diff * 0.618)) / close * 100

    topos, fundos = detectar_topos_fundos(closes)
    tipo_num, tipo_nome, direcao, preco_entrada, preco_referencia = detectar_padrao(topos, fundos)

    # Se o padrao é recente (entrada proxima do ultimo fechamento), usa o ultimo fechamento
    # Senão mantém o preco do padrao para nao distorcer o calculo
    if preco_entrada > 0 and abs(close - preco_entrada) / preco_entrada < 0.15:
        preco_entrada = close

    # Calcular rr com base no padrao
    if direcao == "VENDA" and preco_entrada > 0 and preco_referencia > 0:
        stop_temp = preco_entrada * 1.02
        risco = stop_temp - preco_entrada
        retorno = preco_entrada - preco_referencia
        rr = round(retorno / risco, 2) if risco > 0 else 0
    elif direcao == "COMPRA" and preco_entrada > 0 and preco_referencia > 0:
        stop_temp = preco_entrada * 0.98
        risco = preco_entrada - stop_temp
        retorno = preco_referencia - preco_entrada
        rr = round(retorno / risco, 2) if risco > 0 else 0
    else:
        rr = 0

    return {
        "preco_vs_mm20":  round(preco_vs_mm20, 2),
        "preco_vs_mm50":  round(preco_vs_mm50, 2),
        "preco_vs_mm200": round(preco_vs_mm200, 2),
        "mm20_vs_mm50":   round(mm20_vs_mm50, 2),
        "volume_vs_media": round(volume_vs_media, 2),
        "sombra_sup_pct": round(sombra_sup_pct, 2),
        "sombra_inf_pct": round(sombra_inf_pct, 2),
        "corpo_pct":      round(corpo_pct, 2),
        "dist_fib382":    round(dist_fib382, 2),
        "dist_fib500":    round(dist_fib500, 2),
        "dist_fib618":    round(dist_fib618, 2),
        "rr":             rr,
        "tipo_num":       tipo_num,
        "tipo_nome":      tipo_nome,
        "direcao":        direcao,
        "preco_entrada":  round(preco_entrada, 2),
        "preco_referencia": round(preco_referencia, 2),
    }

def calcular_operacao(direcao, entrada, referencia):
    if direcao == "VENDA":
        stop  = round(entrada * 1.02, 2)
        alvo1 = round(referencia, 2)
        alvo2 = round(alvo1 - (stop - entrada), 2)
        risco   = stop - entrada
        retorno = entrada - alvo1
    elif direcao == "COMPRA":
        stop  = round(entrada * 0.98, 2)
        alvo1 = round(referencia, 2)
        alvo2 = round(alvo1 + (entrada - stop), 2)
        risco   = entrada - stop
        retorno = alvo1 - entrada
    else:
        return None, None, None, None
    rr = round(retorno / risco, 2) if risco > 0 else 0
    return stop, alvo1, alvo2, rr

@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()
    candles = dados.get("candles", [])

    if len(candles) < 30:
        return jsonify({"erro": "Dados insuficientes — minimo 30 candles"}), 400

    feat = calcular_features(candles)

    if feat["tipo_num"] == -1:
        return jsonify({
            "direcao": "INDEFINIDO",
            "tipo_padrao": "SEM PADRAO",
            "mensagem": "Nenhum padrao identificado no periodo visivel",
            "confianca": 0,
            "entrada": None, "stop": None,
            "alvo1": None, "alvo2": None, "rr": None
        })

    X = np.array([feat[c] for c in colunas]).reshape(1, -1)
    prob      = modelo.predict_proba(X)[0][1]
    confianca = round(prob * 100, 1)

    stop, alvo1, alvo2, rr = calcular_operacao(
        feat["direcao"], feat["preco_entrada"], feat["preco_referencia"]
    )

    entrada = feat["preco_entrada"]
    direcao = feat["direcao"]

    # Validacao: alvos devem estar do lado correto da entrada
    if direcao == "VENDA" and (alvo1 is None or alvo1 >= entrada or alvo2 >= alvo1):
        return jsonify({
            "direcao": "INDEFINIDO",
            "tipo_padrao": "SEM PADRAO",
            "mensagem": "Padrao detectado mas alvos inconsistentes com preco atual",
            "confianca": 0,
            "entrada": None, "stop": None,
            "alvo1": None, "alvo2": None, "rr": None
        })

    if direcao == "COMPRA" and (alvo1 is None or alvo1 <= entrada or alvo2 <= alvo1):
        return jsonify({
            "direcao": "INDEFINIDO",
            "tipo_padrao": "SEM PADRAO",
            "mensagem": "Padrao detectado mas alvos inconsistentes com preco atual",
            "confianca": 0,
            "entrada": None, "stop": None,
            "alvo1": None, "alvo2": None, "rr": None
        })

    return jsonify({
        "direcao":     direcao,
        "tipo_padrao": feat["tipo_nome"],
        "entrada":     entrada,
        "stop":        stop,
        "alvo1":       alvo1,
        "alvo2":       alvo2,
        "rr":          rr,
        "confianca":   confianca
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "modelo": "B3 ML v1"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
