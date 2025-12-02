import pandas as pd

from global_config import NODES


def preprocess_latency_txt(file_path, output_csv):
    """
    将txt格式的延迟日志转为标准CSV
    参数：
        file_path: 原始txt路径
        output_csv: 输出csv路径
    """
    times, latencies = [], []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            strs = line.split(',')
            latency = strs[1].split(' ')[-2].strip('(')
            time = strs[0]

            times.append(time)
            latencies.append(latency)

    df = pd.DataFrame({"time": times, "latency": latencies})
    df.to_csv(output_csv, index=False)
    print(f"✅ 已生成标准CSV: {output_csv}")
    print(df.head())

if __name__ == '__main__':
    node_num = len(NODES)
    for i in range(node_num):
        nodeFrom = NODES[i]
        for nodeTo in NODES[i+1:node_num]:
            input_path = f"latency/{nodeFrom}2{nodeTo}.txt"
            output_path = f"latency_csv/{nodeFrom}2{nodeTo}.csv"
            preprocess_latency_txt(input_path, output_path)



