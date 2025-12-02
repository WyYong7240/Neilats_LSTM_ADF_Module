# 1️⃣ 基础镜像
FROM python:3.9-slim

# 2️⃣ 工作目录
WORKDIR /app

# 3️⃣ 拷贝项目文件
COPY . /app

# 4️⃣ 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ 暴露端口
EXPOSE 8080

# 6️⃣ 启动命令
CMD ["python", "predict_app_flask.py"]
