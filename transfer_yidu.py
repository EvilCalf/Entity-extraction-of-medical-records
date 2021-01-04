import os
import pandas as pd


class TransferData:
    def __init__(self):
        self.label_dict = {
            '疾病和诊断': 'DISEASE',
            '影像检查': 'TESTPROC',
            '实验室检验': 'TESTLAB',
            '手术': 'TREATMENT',
            '解剖部位': 'BODY',
            '药物': 'DRUGS',
        }

        self.cate_dict = {
            'O': 0,
            'DISEASE-B': 1,
            'DISEASE-I': 2,
            'TESTPROC-B': 3,
            'TESTPROC-I': 4,
            'TESTLAB-B': 5,
            'TESTLAB-I': 6,
            'BODY-B': 7,
            'BODY-I': 8,
            'DRUGS-B': 9,
            'DRUGS-I': 10,
            'TREATMENT-B': 11,
            'TREATMENT-I': 12,
        }
        self.entity_dirpath = "yidu"
        self.train_filepath = "train/yidu_train.txt"
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.entity_dirpath):
            for file in files:
                json_path = root+"/"+file
                data = pd.read_json(json_path)
                if data.size == 0:
                    continue
                res_dict = {}
                content = data["originalText"][0]
                for i in enumerate(data["entities"]):
                    start = int(i[1]['start_pos'])
                    end = int(i[1]['end_pos'])
                    label = i[1]["label_type"]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start:
                            label_cate = label_id + '-B'
                        else:
                            label_cate = label_id + '-I'
                        res_dict[i] = label_cate
                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')
                    if char != ' ':
                        f.write(char + '\t' + char_label + '\n')
                print("%s 完成！" % json_path)
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
