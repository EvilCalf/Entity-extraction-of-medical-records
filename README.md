# Entity-extraction-of-medical-records

- 病历实体化提取

- 本项目是针对医疗数据，进行命名实体识别。

- - data:已标注的医疗数据，O非实体部分,TREATMENT治疗方式, BODY身体部位, SIGN疾病症状, CHECK医学检查, DISEASE疾病实体（B表示该实体开始，I表示后续字符）
- - data_origin:项目提供的医疗数据，需要转化为目标序列标记集合
- - transfer_data:目标序列化脚本
- - model：训练模型需要的字向量,包含模型文件h5，预训练词向量.bin，以及词表.txt（只有在词表里的词才会被预测为实体类型之一，否则都是O）
- - data_out,需要预测输出的病历文件夹
- - data_out_json,输出的病历实体化提取结果以json形式保存

- python 3.6/keras 2.0.8/tensorflow 1.4/keras_contrib需要使用本项目下的。