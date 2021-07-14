import os
import pandas as pd
import csv
import json
import jsonlines

class TransferData:
    def __init__(self):
        self.label_dict = {
            "疾病": "DISEASE",
            "异常检查结果": "SIGN",
            "边缘": "MARGIN",
            "直径": "DIAMETER",
            "检查过程": "TESTPROC",
            "治疗过程": "TREATMENT",
            "部位": "ANATOMY",
            "性质": "NATURE",
            "形状": "SHAPE",
            "密度": "DENSITY",
            "边界": "BOUNDARY",
            "肺野": "LUNGFIELD",
            "纹理": "TEXTURE",
            "透亮度": "TRANSPARENCY"
        }
        self.cate_dict = {
            "O": 0,
            "DISEASE-B": 1,
            "DISEASE-I": 2,
            "SIGN-B": 3,
            "SIGN-I": 4,
            "MARGIN-B": 5,
            "MARGIN-I": 6,
            "DIAMETER-B": 7,
            "DIAMETER-I": 8,
            "TESTPROC-B": 9,
            "TESTPROC-I": 10,
            "TREATMENT-B": 11,
            "TREATMENT-I": 12,
            "ANATOMY-B": 13,
            "ANATOMY-I": 14,
            "NATURE-B": 15,
            "NATURE-I": 16,
            "SHAPE-B": 17,
            "SHAPE-I": 18,
            "DENSITY-B": 19,
            "DENSITY-I": 20,
            "BOUNDARY-B": 21,
            "BOUNDARY-I": 22,
            "LUNGFIELD-B": 23,
            "LUNGFIELD-I": 24,
            "TEXTURE-B": 25,
            "TEXTURE-I": 26,
            "TRANSPARENCY-B": 27,
            "TRANSPARENCY-I": 28
        }
        self.entity_dirpath = "data/sict"
        self.train_filepath = "train/sict_train.txt"
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.entity_dirpath):
            for file in files:
                # with open(root+"/"+file, "r", encoding='utf-8') as f:
                #     reader = csv.reader(f)
                #     data = list(reader)
                # if len(data) == 0:
                #     continue
                data=[]
                with open(root+"/"+file, "r+", encoding="utf8") as f1:
                    for item in jsonlines.Reader(f1):
                        data.append(item)
                for i,strs in enumerate(data):
                    res_dict = {}
                    for j,raw in enumerate(strs['label']):
                        start = int(raw[0])
                        end = int(raw[1])
                        label_id = raw[2]
                        for i in range(start, end):
                            if i == start:
                                label_cate = 'B-'+label_id
                            else:
                                label_cate = 'I-'+label_id
                            res_dict[i] = label_cate
                    for indx, char in enumerate(strs['data']):
                        char_label = res_dict.get(indx, 'O')
                        word_list = ['。']
                        if char in word_list:
                            char_label = 'O'
                        if char != ' ':
                            f.write(char + '\t' + char_label + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
