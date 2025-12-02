import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 归一化
minMaxScaler = MinMaxScaler()     # 对数据进行归一化，缩放到0-1区间，防止数值过大影响神经网络训练稳定性
robustScaler = RobustScaler()
standardScaler = StandardScaler()
SQL_LEN = 30    # 定义时间窗口长度，即每次输入LSTM的连续时间步为30，例如用前30秒的延迟值预测第31秒的延迟
EPOCHS = 50     # 训练轮数

# 构造时序样本
def create_sequences(data, seq_len=30):
    xs, ys = [], []
    # 30个一组
    for i in range(len(data) - seq_len):
        # x形状为(seq_len, 1), y形状为(1)
        x = data[i:i+seq_len]   # 取i到i+seq_len-1的数据作为输入
        y = data[i+seq_len]     # 取第i+seq_len个数组作为标签
        xs.append(x)
        ys.append(y)
    # xs形状(seq_num, seq_len, 1), y形状为(seq_num, 1)
    return np.array(xs), np.array(ys)

def data_prepare(path):
    # 读取数据
    # 从CSV读取数据，并将time列解析为datetime类型
    df = pd.read_csv(path, parse_dates=['time'])
    # 按时间排序，否则打乱依赖关系
    df = df.sort_values('time')


    print("Latency min:", df['latency'].min())
    print("Latency max:", df['latency'].max())
    print("Latency mean:", df['latency'].mean())
    print("Latency std:", df['latency'].std())

    # 只取latency列,并reshape为2维，因为MinMaxScaler要求输入为2D:(N,1)
    values = df['latency'].values.reshape(-1, 1)

    # 使用MinMaxScaler将latency数据归一化到0-1区间;这里使用了fit_transform,如果有多个文件，建议只对第一个fit，后面用transform,形状不变
    # scaled = minMaxScaler.fit_transform(values)
    # scaled = robustScaler.fit_transform(values)
    scaled = standardScaler.fit_transform(values)

    # 生成时序样本
    X, y = create_sequences(scaled, SQL_LEN)
    return X, y

# 自定义pytorch数据集
class TimeSeriesDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_test_partition(X, y):
    # 划分训练集/测试集
    # 按80%训练，20%测试划分数据，时间序列不能随机打乱，所以是前80%训练
    train_size = int(0.8 * len(X))
    train_dataset = TimeSeriesDataSet(X[:train_size], y[:train_size])
    test_dataset = TimeSeriesDataSet(X[train_size:], y[train_size:])

    # 训练集每个epoch打乱样本顺序，有助于泛化
    # x_batch形状为(batch_size, seq_len, 1), y_batch形状为(batch_size, 1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 测试集保持时间顺序以便于后续绘图对比
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

# 定义LSTM模型
class LSTMModel(nn.Module):
    # 继承nn.Module，默认输入特征维度latency只有1，隐藏层大小64，LSTM层数2
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建一个LSTM层，输入形状为(batch，seq_len，features);dropout=0.2在除了最后一层外的所有层之间添加dropout，防止过拟合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        # 全连接层，即输出层，将LSTM最后一个时间步的输出映射到预测值，通常是1维
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 手动初始化LSTM的初始隐藏状态h0和细胞状态c0
        # 形状(num_layers, batch_size, hidden_size)
        # 使用.to(x.device)确保它们与输入张量在同一个设备
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播通过LSTM，得到所有时间步的输出out，形状为(batch_size, seq_len, hidden_size)
        out, _ =self.lstm(x, (h0, c0))
        # 只取最后一个时间步的输出，因为这是单步预测，然后送入全连接层输出最终预测
        # out[:,-1,:]形状为(batch_size, hidden_size)
        # 最终out形状为(batch_size, output_size)
        out = self.fc(out[:, -1, :])
        return out

def train_and_test_model(train_loader, test_loader):
    # 训练模型
    # 自动选择设备，有cuda用cuda，没有用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型并移动到指定设备
    model = LSTMModel().to(device)

    # 定义损失函数均方误差和优化器Adam
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

    torch.save(model.state_dict(), "model/lstm_single_step.pth")
    torch.save(standardScaler, "model/scaler_single_step.pkl")

    #测试与预测
    # 关闭Dropout，启动评估模式
    model.eval()
    preds, trues = [], []

    # 禁用梯度计算，节省内存
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            preds.append(y_pred.cpu().numpy())  # 将GPU张量转回Numpy
            trues.append(y_batch.numpy())

        # 将多个batch的预测结果拼接成完整数组
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

    # 反归一化;使用之前拟合的scaler将归一化的预测值还原为原始尺度，便于解释和绘图
    # preds_inv = minMaxScaler.inverse_transform(preds)
    # trues_inv = minMaxScaler.inverse_transform(trues)
    # preds_inv = robustScaler.inverse_transform(preds)
    # trues_inv = robustScaler.inverse_transform(trues)
    preds_inv = standardScaler.inverse_transform(preds)
    trues_inv = standardScaler.inverse_transform(trues)
    return preds_inv, trues_inv

def consequence_show(preds_inv, trues_inv):
    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.plot(trues_inv, label="True")
    plt.plot(preds_inv, label="Predicated")
    plt.title("LSTM Latency Predication")
    plt.xlabel("Time Step")
    plt.ylabel("Latency")
    plt.legend()
    plt.show()

def main():
    X, y = data_prepare("latency_csv/master2node1.csv")
    train_loader, test_loader = train_test_partition(X, y)
    preds_inv, trues_inv = train_and_test_model(train_loader, test_loader)
    consequence_show(preds_inv, trues_inv)


if __name__ == '__main__':
    # preprocess_latency_txt("latency/master2node1.txt", "latency_csv/master2node1.csv")
    main()
    # print("PyTorch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA version:", torch.version.cuda)
    # print("Current device:", torch.cuda.current_device())
    # print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
