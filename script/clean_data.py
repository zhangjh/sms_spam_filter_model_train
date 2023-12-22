# 原始数据清理，将数据处理成已分词后可以直接训练的数据
# data/clean下的文件已经预处理好
import pandas as pd
import jieba
import zhconv as zhconv


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

def get_label(x):
    splits = x.split("    ")
    return splits[0]
def clean(x):
    splits = x.split("    ")
    x = splits[1]
    segs = jieba.cut(x, cut_all=False)
    result = []
    stopwords = load_stopwords(stopwords_path)
    for seg in segs:
        if seg not in [" ", "\t", "\n", "\r", "\u3000", "\xa0"]:
            # 去除既不是汉字也不是英文字母
            if not seg.isalpha():
                continue
            if seg not in stopwords and seg != "":
                if not result.__contains__(seg):
                    result.append(zhconv.convert(seg.strip(), "zh-hans"))
    return " ".join(result)
stopwords_path = "../data/scu_stopwords.txt"

data = pd.read_csv("../data/sms_pub.new", header=None,
                   encoding="utf-8-sig", sep="\t")
# print(data.head())
# data[0] = data[0].apply(lambda x: x.replace("\\s", ""))
data[1] = data[0].apply(clean)
data[0] = data[0].apply(get_label)
# print(data.head())
data.to_csv("./data/clean/sms_pub.csv", header=False, index=False, encoding="utf-8-sig", sep="\t")
