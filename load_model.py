import torch
import joblib
from lstm_multi_step import LSTMModel, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

def load_trained_model(model_path, scaler_path):
    # 实例化模型：参数需要与训练时一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型并移动到指定设备
    model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 模型移动到对应设备
    model.to(device)
    model.eval()

    # 加载scaler
    scaler = joblib.load(scaler_path)
    print(f"模型和Scaler加载成功，当前设备{device}")
    return model, scaler
