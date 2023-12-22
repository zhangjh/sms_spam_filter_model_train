# python下验证onnx模型使用
import onnx
import onnxruntime

onnx_model = onnx.load("./model/model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

msg = ["【国金证券】让闲钱自动参与国债逆回购，试试余额理财条件单，今日操作或连享3日收益！登录国金佣金宝App"]
for i in range(len(msg)):
    ort_inputs = {ort_session.get_inputs()[0].name: [msg[i]]}
    ort_outs = ort_session.run(None, ort_inputs)

    print(ort_outs)