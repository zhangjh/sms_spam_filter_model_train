# 训练和导出模型
import os
import threading
import time

import pandas as pd
from onnxconverter_common import StringTensorType
from skl2onnx import convert_sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(data_path):
    encoding = "utf-8-sig"
    model_data = data_path
    chunksize = 1000
    chunks = []
    for chunk in pd.read_csv(model_data,
                             sep="\t",
                             chunksize=chunksize,
                             encoding=encoding,
                             header=None,
                             usecols=[0, 1]):
        chunks.append(chunk)
    handled_data = pd.concat(chunks)
    handled_data.dropna(axis=0, how="any", inplace=True)
    return handled_data


def fit_pipeline(pipeline, X_train, y_train):
    print(threading.current_thread().name)
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == '__main__':
    model_data = "./data/clean/all"
    handled_data = read_data(model_data)
    # 文本
    X = handled_data[1]
    # 标签
    y = handled_data[0].astype('int')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=False)),
        ('clf', LogisticRegression(verbose=0))
    ])

    start_time = time.time()
    trained_pipeline = fit_pipeline(pipeline, X_train, y_train)
    cost = time.time() - start_time
    print(cost)

    # 在测试集上测试模型
    y_pred = trained_pipeline.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # 用例验证
    y_pred = trained_pipeline.predict(["【国金证券】让闲钱自动参与国债逆回购，试试余额理财条件单，今日操作或连享3日收益！登录国金佣金宝App"])
    print(y_pred)

    # 定义模型的输入类型
    initial_type = [('message', StringTensorType([1]))]

    # 转换模型为ONNX格式
    onx = convert_sklearn(trained_pipeline, initial_types=initial_type)
    # 查看模型的信息

    # 保存模型
    mkdir("../model")
    with open("../model/model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
