---
tags:
  - neilats_refactor
  - LSTM
  - ADF
  - Flask
---
# Neilats_LSTM_ADF_Module

该项目的作用是为Neilats自定义调度插件提供另外的两个功能模块

1. **用于评判节点间网络稳定性的ADF得分计算模块**

   在该项目中实现，是因为如果要在Neilats自定义调度插件中实现ADF平稳性检验计算模块，没有原生的Go包可以使用，在调度插件中自己实现会让调度插件变得沉重

   因此还是考虑将其作为一个HTTP服务在集群上部署，通过服务的方式给Neilats自定义调度插件访问使用

   [Neilats自定义调度插件](https://github.com/WyYong7240/neilats-refactor-scheduler-plugins/tree/master)

2. **用于预测节点间未来网络延迟的LSTM延迟预测和未来网络评分模块**

   LSTM预测服务当然是无法在Go中实现了，因此还是在Python中实现，通过服务的形式提供给Neilats

   而评分模块，虽然实现简单，但是由于评分还是要使用LSTM模块预测出来的未来网络延迟数据，所以还是一起评分再返回给调度插件比较好。

# 使用方法

   ## 程序启动

   1. 通过Python程序直接启动

      ~~~sh
      python3 predict_app_flask.py
      ~~~

      默认端口是`8080`

   2. 构建为镜像启动

      ~~~sh
      docker build -t lstm-adf-module:v1.0 .
      ~~~

   ## 访问端口

   1. `http://<IP>:<Port>/predict/master2node1`

      用于预测节点master和node1之间的网络延迟，并返回未来网络链路得分

   2. `http://<IP>:<Port>/predict/master2node2`

      用于预测节点master和node2之间的网络延迟，并返回未来网络链路得分

   3. `http://<IP>:<Port>/predict/node12node2`

      用于预测节点node1和node2之间的网络延迟，并返回未来网络链路得分

   4. `http://<IP>:<Port>/adf_score`

      用于计算节点间的网络稳定性得分

   ## 测试用例

   1. 未来网络得分

      ~~~sh
      curl -X POST http://127.0.0.1:8080/predict/master2node1 -H "Content-Type: application/json" -d "{\"latency\": [0.382, 0.370, 0.389, 0.434, 0.331,0.355, 0.361, 0.380, 0.311, 0.374, 0.473, 0.376, 0.357, 0.356, 0.329, 0.375, 0.373, 0.372, 0.359, 0.357, 0.435, 0.396, 0.354, 0.352, 0.362,0.364, 0.495, 0.395, 0.343, 0.347], \"sla_time\":0.35}"
      ~~~

   2. ADF网络稳定性得分

      ~~~sh
      curl -X POST http://127.0.0.1:8080/adf_score -H "Content-Type: application/json" -d "{\"latency\": [0.382, 0.370, 0.389, 0.434, 0.331,0.355, 0.361, 0.380, 0.311, 0.374, 0.473, 0.376, 0.357, 0.356, 0.329, 0.375, 0.373, 0.372, 0.359, 0.357, 0.435, 0.396, 0.354, 0.352, 0.362,0.364, 0.495, 0.395, 0.343, 0.347]}"
      ~~~

# 模块介绍

## LSTM时间序列预测模型训练与测试

相关技术文档:https://github.com/WyYong7240/ObsidianNotes/tree/main/PyTorch

* 涉及到的文件和目录

  1. latency_csv

     用于存储节点间的预处理完毕的延迟数据

  2. model

     用于存储不同节点间已经训练好的延迟预测模型和Scaler

  3. earlyStopping.py

     用于模型训练早停策略的代码，但是没有被使用

  4. lstm_multi_step.py

     LSTM延迟单步预测代码，利用过去的30步预测未来1步

  5. lstm_single_step.py

     LSTM延迟多步预测代码，利用过去的30步预测未来的30步

  6. process_txt_to_csv.py

     用于将节点间原始的延迟txt数据，预处理为CSV文件。原始延迟数据文件已经被删除，仅留存该预处理代码

## LSTM延迟预测服务模块

相关技术文档:https://github.com/WyYong7240/ObsidianNotes/tree/main/PyTorch

* 涉及到的文件和目录

  1. model

     预测服务从该目录中加载预测模型和Scaler

  2. load_model.py

     预测服务使用的模型加载代码，用于在启动时，准备不用节点的预测模型

  3. predict_app_flask.py

     预测服务代理模块，使用flask实现

## ADF网络稳定性得分计算、未来网络链路得分计算模块

* 设计到的文件和目录

  1. adf_module.py

     包含了ADF网络稳定性得分计算模块，未来网络链路得分计算模块

  2. predict_app_flask.py

     得分计算服务代理模块，使用flask实现