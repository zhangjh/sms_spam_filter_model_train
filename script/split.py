import os

import pandas as pd
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from zhconv import zhconv

import jieba


def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    stopwords += list(hanzi_biaodian()) + list(english_biaodian()) + list(hanzi_symbols())
    return stopwords


def hanzi_biaodian():
    return "、，。！？；：“”‘’（）【】《》【】｛｝〔〕〈〉〖〗「」『』〃—…－～·＠＃％＆＊＋－／＜＝＞＼＾＿｀｜￣"


def english_biaodian():
    return "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

def hanzi_symbols():
    return "①②③④⑤⑥⑦⑧⑨"

def jieba_cut(x):
    segs = jieba.cut(x, cut_all=False)
    # 去除停顿词
    result = []
    stopwords = load_stopwords(stopwords_path)
    for seg in segs:
        # 去除既不是汉字也不是英文字母
        if not seg.isalpha():
            continue
        if seg not in stopwords and seg != "":
            if not result.__contains__(seg):
                result.append(zhconv.convert(seg.strip(), "zh-hans"))
    return " ".join(result)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


"""
   对文件进行jieba分词处理
   参数：
   file_name -- 文件名
   返回值：
   无
"""


def split_file(file_name, fenci_data):
    if os.path.isfile(fenci_data):
        return
    df = pd.read_table(file_name, header=None, sep="\t", encoding="utf-8-sig")
    df[1] = df[1].apply(jieba_cut)
    # df.drop(columns=[1, 2], inplace=True)
    df.to_csv(fenci_data, header=False, index=False, encoding="utf-8-sig", sep="\t")


# "./model/model.onnx"
def save_model(save_path, X_train, clf):
    # 转换模型为ONNX格式
    initial_type = [('message', FloatTensorType([None, X_train.shape[1]]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    mkdir("tag/model")
    with open(save_path, "wb") as f:
        f.write(onx.SerializeToString())


def train_model(model_data):
    # 读取最终处理好（预先打标）的数据
    chunksize = 1000
    chunks = []
    for chunk in pd.read_csv(model_data, chunksize=chunksize, header=None, sep="\t",
                             encoding="utf-8-sig", na_values=[], keep_default_na=False):
        # chunk = chunk.iloc[:, :2]
        chunks.append(chunk.iloc[:, :2])
    handled_data = pd.concat(chunks)
    # 提取文本
    X = handled_data[1]
    # 提取标签
    y = handled_data[0]

    tfidVectorizer = TfidfVectorizer(lowercase=False)
    X = tfidVectorizer.fit_transform(X)

    ## 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, y_train)

    # 在测试集上评估性能
    y_pred = clf.predict(X_test)
    print("准确率：", sklearn.metrics.accuracy_score(y_test, y_pred))

    # 在验证集上验证模型的有效性
    data_train = pd.read_csv("../data/clean/github.csv", header=None, sep="\t",
                             encoding="utf-8-sig", na_values=[], keep_default_na=False)
    train_texts = data_train[1]
    train_label = data_train[0]
    train_texts = tfidVectorizer.transform(train_texts)
    X_train, X_test, y_train, y_test = train_test_split(train_texts, train_label, test_size=0.2, random_state=42)

    y_pred = clf.predict(X_test)
    print(y_pred)
    print("准确率：", sklearn.metrics.accuracy_score(y_test, y_pred))

    save_model("../model/model.onnx", X_train, clf)

    new_sms_features = tfidVectorizer.transform(
        ["【嘉实基金】2023已接近尾声，小编收集了今年投资者们普遍关注的问题：ETF基金的份额和规模为什么会在今年迎来大爆发？怎么能够查到在各家基金公司或平台买的所有基金产品？以上问题的答案尽在今晚七点半直播间，点击  https://t.chinaharvest.com/rRwfyiQ  （官方信息 谨防失效）观看！投资需谨慎。拒收请回复R",
        "【阿里云】尊敬的njhxzhangjihong@126.com，阿里云小程序及酷应用已经上架钉钉，您可以便捷地管理云资源、续费云产品，还可以将告警消息实时推送到群聊或指定员工，降本提效。立即体验：https://c.tb.cn/F3.bgOl5 T",
         "【国金证券】让闲钱自动参与国债逆回购，试试余额理财条件单，今日操作或连享3日收益！登录国金佣金宝App"
            "-交易-智能条件单-余额理财条件单，立即体验。低风险，自动下单，资金次日可用于交易。详情请咨询服务人员>>https://g.gjyjb.com/rd/oLJ8 投资有风险，拒收请回复R"])
    prediction = clf.predict(new_sms_features.toarray())
    print(prediction)


# 原始已打标短信数据文件（两列：label，content）
ori_data = "./data/github.csv"
# 经过jieba分词后的数据文件，只保留分词后的内容（两列：label，content_jieba）
fenci_data = "./data/github.8k"
stopwords_path = "../data/scu_stopwords.txt"

## jieba分词
# split_file("./personal.txt", "./jieba/personal.csv")

## 训练模型
train_model("./data/clean/personal.csv")


