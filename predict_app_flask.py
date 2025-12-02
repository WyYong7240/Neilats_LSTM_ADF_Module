import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from load_model import load_trained_model
from lstm_multi_step import SEQ_LEN
from global_config import NODES
from adf_module import adf, def_future_score

app = Flask(__name__)
CORS(app)

MODEL_SCALER_PATH = "model/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = dict()
SCALERS = dict()

def get_input_tensor(scaler_name):
    data = request.get_json()
    if not data or "latency" not in data:
        return jsonify({"error": "Missing 'latency' field"}), 400
    # 提取输入的数据
    values = np.array(data["latency"], dtype=float).reshape(-1, 1)
    if len(values) != 30:
        return jsonify({"error": "Latency length is not 30"}), 400
    # 缩放
    scaled_data = SCALERS[scaler_name].transform(values).reshape(1, SEQ_LEN, 1)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(DEVICE)

    # 获取sla_time字段
    if "sla_time" not in data:
        return jsonify({"error": "Missing 'sla_time' field"}), 400
    sla_time = float(data["sla_time"])

    return input_tensor, sla_time

@app.route('/predict/master2node1', methods=['POST'])
def predict_master2node1():
    try:
        # 获取数据
        input_tensor, sla_time = get_input_tensor("master2node1")
        # 推理
        with torch.no_grad():
            pred_scaled = MODELS["master2node1"](input_tensor)
            pred_scaled = pred_scaled.cpu().numpy()
        # 反归一化
        pred_inv = SCALERS["master2node1"].inverse_transform(pred_scaled.reshape(-1, 1))

        # 计算未来通信链路得分
        print("master2node1 predict latency:")
        print(pred_inv.tolist())
        future_score = def_future_score(pred_inv.tolist(), sla_time)
        print("master2node1 future_score:")
        print(future_score)
        return jsonify({"future_score": future_score})

        # return jsonify({"prediction": pred_inv.tolist()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/predict/master2node2', methods=['POST'])
def predict_master2node2():
    try:
        # 获取数据
        input_tensor, sla_time = get_input_tensor("master2node2")
        # 推理
        with torch.no_grad():
            pred_scaled = MODELS["master2node2"](input_tensor)
            pred_scaled = pred_scaled.cpu().numpy()
        # 反归一化
        pred_inv = SCALERS["master2node2"].inverse_transform(pred_scaled.reshape(-1, 1))

        # 计算未来通信链路得分
        print("master2node2 predict latency:")
        print(pred_inv.tolist())
        future_score = def_future_score(pred_inv.tolist(), sla_time)
        print("master2node2 future_score:")
        print(future_score)
        return jsonify({"future_score": future_score})

        # return jsonify({"prediction": pred_inv.tolist()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/predict/node12node2', methods=['POST'])
def predict_node22node1():
    try:
        # 获取数据
        input_tensor, sla_time = get_input_tensor("node12node2")
        # 推理
        with torch.no_grad():
            pred_scaled = MODELS["node12node2"](input_tensor)
            pred_scaled = pred_scaled.cpu().numpy()
        # 反归一化
        pred_inv = SCALERS["node12node2"].inverse_transform(pred_scaled.reshape(-1, 1))

        # 计算未来通信链路得分
        print("node12node2 predict latency:")
        print(pred_inv.tolist())
        future_score = def_future_score(pred_inv.tolist(), sla_time)
        print("node12node2 future_score:")
        print(future_score)
        return jsonify({"future_score": future_score})

        # return jsonify({"prediction": pred_inv.tolist()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/adf_score', methods=['POST'])
def adf_score_compute():
    data = request.get_json()
    if not data or "latency" not in data:
        return jsonify({"error": "Missing 'latency' field"}), 400
    print(data["latency"])
    adf_score = adf(data["latency"])
    print("adf_score:")
    print(adf_score)
    return jsonify({"adf_score": adf_score})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({"error": "Not running with the Werkzeug Server"}), 500
    func()
    return jsonify({"message": "Server shutting down..."})

if __name__ == '__main__':
    node_num = len(NODES)
    for i in range(node_num):
        nodeFrom = NODES[i]
        for nodeTo in NODES[i+1:node_num]:
            model_name = f"{nodeFrom}2{nodeTo}"
            model_path = MODEL_SCALER_PATH + f"lstm_{model_name}.pth"
            scaler_path = MODEL_SCALER_PATH + f"scaler_{model_name}.pkl"
            model, scaler = load_trained_model(model_path, scaler_path)
            MODELS[model_name] = model
            SCALERS[model_name] = scaler
            print(f"{model_name}的Model和Scaler加载成功！")
    app.run(host='0.0.0.0', port=8080, debug=False)