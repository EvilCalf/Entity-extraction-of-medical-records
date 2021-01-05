# Entity Extraction of Medical Mecords

- 病历实体化提取

- 本项目是针对医疗数据，进行命名实体识别。

- - data:已标注的医疗数据，O非实体部分,TREATMENT手术, BODY解剖部位, TESTLAB实验室检验, TESTPROC影像检查, DISEASE疾病和诊断（B表示该实体开始，I表示后续字符）
- - yidu:yidu病历实体数据集
- - transfer_yidu.py:把yidi病历实体数据集转化成BIEO格式
- - model：训练模型需要的字向量,包含模型文件h5，预训练词向量.bin，以及词表.txt（训练自动生成，只有在词表里的字才会被预测为实体类型之一，否则都是O）
- - data_out,需要预测输出的病历文件夹
- - data_out_json,输出的病历实体化提取结果以json形式保存

- python 3.6/keras 2.0.8/tensorflow 1.4/keras_contrib需要使用本项目下的。