from statsmodels.tsa.stattools import adfuller

def adf(latency_data):
    # 传入的延迟数据就是已经预处理好的延迟数据，长度为预定义的ADF_LEN
    adf_result = adfuller(latency_data)
    p_value = adf_result[1]
    t_value = adf_result[0]

    if p_value <= 0.01:
        adf_score = 10 * (adf_result[4]["1%"] - t_value)
    elif p_value > 0.01 and p_value <= 0.05:
        adf_score = 10 * (adf_result[4]["5%"] - t_value)
    elif p_value > 0.05 and p_value <= 0.1:
        adf_score = 10 * (adf_result[4]["10%"] - t_value)
    else:
        adf_score = 0
    return adf_score

def def_future_score(predict_latency, sla_time):
    predict_score = 0
    # print(predict_latency)
    for latency in predict_latency:
        # print(latency)
        predict_score = predict_score + (sla_time - float(latency[0])) / float(latency[0])
    return predict_score


