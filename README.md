# sms_spam_filter_model_train
train a model to classfier sms spam or not

最终可以生成一个model.onnx格式的模型，训练数据除互联网上公开的数据外，还有个人打标好的近8k条短信数据。
出于隐私保护，这部分数据并未入库。这部分数据打标分为三类，0-正常，1-垃圾，2-未确定

本来都目的是为了根据这个模型做一个Android APP，用来过滤自己的短信，但是在后续的Android APP开发过程中发现，Google在Android4.4之后禁止了用户应用对短信的拦截。

因此该训练模型也没办法实际应用起来了，本仓库仅作为一个模型训练的demo，里面关于onnx模型导出及使用，jieba分词器的使用和数据预处理部分还是值得记录供后续的模型训练过程中借鉴使用的。