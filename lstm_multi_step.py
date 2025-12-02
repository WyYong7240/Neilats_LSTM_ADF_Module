import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from global_config import NODES

# 归一化
# scaler = MinMaxScaler()     # 对数据进行归一化，缩放到0-1区间，防止数值过大影响神经网络训练稳定性
# scaler = RobustScaler()
scaler = StandardScaler()
SEQ_LEN = 30    # 定义时间窗口长度，即每次输入LSTM的连续时间步为30，例如用前30秒的延迟值预测第31秒的延迟
PRED_LEN = 30
EPOCHS = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 0.0003

# 构造训练集时序样本
def create_train_multi_step_sequences(data, seq_len=30, pred_len=30):
    xs, ys = [], []
    # 30个一组
    for i in range(len(data) - seq_len - pred_len):
        # 这里，每个x都是一个形状为(seq_len,1)的二维数组;y也同理
        x = data[i:i+seq_len]   # 取i到i+seq_len-1的数据作为输入
        y = data[i+seq_len:i+seq_len+pred_len]     # 取第i+seq_len个数组作为标签
        xs.append(x)
        ys.append(y)
        # 因此，xs的形状是( len(data)-seq_len-pred_len, seq_len, 1)
        # ys的形状是( len(data)-seq_len-pred_len, pred_len, 1)
        # 该3D形状是用于序列模型，例如LSTM、GRU的标准的3D形状，分别代表(样本数，时间步长，特征数)
    return np.array(xs), np.array(ys)

# 构造测试集时序样本，使得没有重复预测点
def create_test_multi_step_sequences(data, seq_len=30, pred_len=30):
    xs, ys = [], []
    # 30个一组
    n_blocks = (len(data) - pred_len) // seq_len
    for i in range(n_blocks):
        # 这里，每个x都是一个形状为(seq_len,1)的二维数组;y也同理
        x = data[i*seq_len:(i+1)*seq_len]
        y = data[(i+1)*seq_len:(i+1)*seq_len+pred_len]
        xs.append(x)
        ys.append(y)
        # 因此，xs的形状是( len(data)-seq_len-pred_len, seq_len, 1)
        # ys的形状是( len(data)-seq_len-pred_len, pred_len, 1)
        # 该3D形状是用于序列模型，例如LSTM、GRU的标准的3D形状，分别代表(样本数，时间步长，特征数)
    return np.array(xs), np.array(ys)

def data_prepare_and_partition(path):
    # 读取数据
    # 从CSV读取数据，并将time列解析为datetime类型
    df = pd.read_csv(path, parse_dates=['time'])
    # 按时间排序，否则打乱依赖关系
    df = df.sort_values('time')

    # 判断是要训练集数据还是测试集数据,如果是训练集，取前80%，测试集取后20%
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    print("Train:")
    print("Latency min:", train_df['latency'].min())
    print("Latency max:", train_df['latency'].max())
    print("Latency mean:",train_df['latency'].mean())
    print("Latency std:", train_df['latency'].std())
    print("Test:")
    print("Latency min:", test_df['latency'].min())
    print("Latency max:", test_df['latency'].max())
    print("Latency mean:",test_df['latency'].mean())
    print("Latency std:", test_df['latency'].std())

    # 只取latency列,并reshape为2维，因为MinMaxScaler要求输入为2D
    # 这里，将latency重塑为一个二维数组，如果原始有N行，那么values的形状是(N,1)，即N行1列
    # 参数含义：-1是占位符，表示自动计算这个维度的大小； 1表示第二维度列数必须是1
    train_values = train_df['latency'].values.reshape(-1, 1)
    test_values = test_df['latency'].values.reshape(-1, 1)

    # 经过这一步，形状不变
    # 使用MinMaxScaler将latency数据归一化到0-1区间;这里使用了fit_transform,如果有多个文件，建议只对第一个fit，后面用transform
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.fit_transform(test_values)

    return train_scaled, test_scaled

# 自定义pytorch数据集
class TimeSeriesDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_train_test_loader(train_scaled, test_scaled):
    # 生成时序样本
    train_X, train_y = create_test_multi_step_sequences(train_scaled, SEQ_LEN, PRED_LEN)
    test_X, test_y = create_test_multi_step_sequences(test_scaled, SEQ_LEN, PRED_LEN)

    # 构造数据集,训练集测试集的划分已经划分完了
    train_dataset = TimeSeriesDataSet(train_X, train_y)
    test_dataset = TimeSeriesDataSet(test_X, test_y)

    # 训练集每个epoch打乱样本顺序，有助于泛化
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 测试集保持时间顺序以便于后续绘图对比
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 该数据迭代器，返回的数据形状是:
    # X_batch:(batch_size, seq_len, features_len)即(32, 30, 1)
    # y_batch:(batch_size, pred_len, features_len)即(32, 30, 1)
    return train_loader, test_loader

# 定义LSTM模型
# LSTM模型期望输入是一个3D张量，形状为(batch_size, seq_len, input_size)也就是序列样本、时间步长、特征数量
class LSTMModel(nn.Module):
    # 继承nn.Module，默认输入特征维度latency只有1，隐藏层大小64，LSTM层数2
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=30, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建一个LSTM层，输入形状为(batch，seq_len，features);dropout=0.2在除了最后一层外的所有层之间添加dropout，防止过拟合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 另外添加一个全连接隐藏层+ReLu
        # self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc_dropout = nn.Dropout(dropout)

        # 全连接层，即输出层，将LSTM最后一个时间步的输出映射到预测值，通常是1维
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # 手动初始化LSTM的初始隐藏状态h0和细胞状态c0
        # 形状(num_layers, batch_size, hidden_size)
        # 使用.to(x.device)确保它们与输入张量在同一个设备
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # 前向传播通过LSTM，得到所有时间步的输出out，形状(bath_size, seq_len, hidden_size)
        lstm_out, _ =self.lstm(x, (h0, c0))
        # 只取最后一个时间步的输出，因为这是单步预测，然后送入全连接层输出最终预测; 形状(batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # 表示':, -1, :'表示，第一维度取全部，第二维度取最后一个，第三维度取全部

        # 取最后一个时间步后，经过全连接隐藏层再到Relu，再到dropout，最后输出
        # fc_hidden_out = self.fc_hidden(last_hidden)
        # relu_out = self.relu(fc_hidden_out)
        # fc_dropout_out = self.fc_dropout(relu_out)

        # 全连接层输出未来pred_len个预测值,形状是(batch_size, output_size)
        predictions = self.fc(last_hidden)
        # 增加一个维度，变为3D,形状为(batch_size, output_size, 1)
        # unsqueeze(-1)表示在最后一个维度后面插入一个大小为1的新维度
        predictions = predictions.unsqueeze(-1)
        return predictions

def train_and_test_model(train_loader, test_loader, model_name):
    # 训练模型
    # 自动选择设备，有cuda用cuda，没有用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型并移动到指定设备
    model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout= DROPOUT).to(device)

    # 定义损失函数均方误差和优化器Adam
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # 启动Dropout/BatchNorm等训练行为
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # 将每一批数据移动到设备上
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()   # 清除梯度
            y_pred = model(X_batch) # 前向传播

            loss = criterion(y_pred, y_batch)   # 计算损失
            loss.backward()     # 反向传播
            optimizer.step()    # 更新参数

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), f"model/lstm_{model_name}.pth")
    # torch.save(standardScaler, "model/standardScaler.pkl")
    joblib.dump(scaler, f"model/scaler_{model_name}.pkl")

    #测试与预测
    # 关闭Dropout，启动评估模式
    model.eval()
    preds, trues = [], []

    # 禁用梯度计算，节省内存
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            preds.append(y_pred.cpu().numpy())  # 将GPU张量转回Numpy
            trues.append(y_batch.cpu().numpy())

        # 将多个batch的预测结果拼接成完整数组
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

    # 反归一化;使用之前拟合的scaler将归一化的预测值还原为原始尺度，便于解释和绘图
    preds_2d = preds.reshape(-1, 1)
    trues_2d = trues.reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds_2d)
    trues_inv = scaler.inverse_transform(trues_2d)
    return preds_inv, trues_inv

def consequence_show(preds_inv, trues_inv, model_name):
    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.plot(trues_inv, label="True (30 future steps")
    plt.plot(preds_inv, label="Predicated (30 future steps)")
    plt.title("LSTM Latency Predication Multi Step 30")
    plt.xlabel("Time Step")
    plt.ylabel("Latency")
    plt.figtext(0.2, 0.02, f"model_name={model_name}, hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, dropout={DROPOUT}, Lr={LR}, strategic=2*Test")
    plt.legend()
    plt.show()

def train_get_model_scaler(node_from, node_to):
    model_name = f"{node_from}2{node_to}"
    csv_path = f"latency_csv/{model_name}.csv"
    train_scaled, test_scaled = data_prepare_and_partition(csv_path)
    train_loader, test_loader = build_train_test_loader(train_scaled, test_scaled)
    preds_inv, trues_inv = train_and_test_model(train_loader, test_loader, model_name)
    consequence_show(preds_inv, trues_inv, model_name)
    print(f"Model And Scaler:{model_name} train complete! Save as file!")


if __name__ == '__main__':
    node_num = len(NODES)
    for i in range(node_num):
        nodeFrom = NODES[i]
        for nodeTo in NODES[i+1:node_num]:
            train_get_model_scaler(nodeFrom, nodeTo)
    # print("PyTorch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA version:", torch.version.cuda)
    # print("Current device:", torch.cuda.current_device())
    # print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
