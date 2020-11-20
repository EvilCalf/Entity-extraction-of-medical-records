import os
import json

class TransferData:
    def __init__(self):
        self.label_dict = {
            '影像检查': 'CHECK',
            '实验室检验': 'CHECK',
            '疾病和诊断': 'DISEASE',
            '手术': 'TREATMENT',
            '解剖部位': 'BODY',
            '药物': 'TREATMENT'}

        self.cate_dict = {
            'O': 0,
            'TREATMENT-I': 1,
            'TREATMENT-B': 2,
            'BODY-B': 3,
            'BODY-I': 4,
            'CHECK-B': 5,
            'CHECK-I': 6,
            'DISEASE-B': 7,
            'DISEASE-I': 8,
            'SIGN-B': 9,
            'SIGN-I': 10,
        }
        self.entity_dirpath = "yidu"
        return

    def transfer(self):
        with open('train/yidu.txt', encoding="utf-8", mode="w+") as f:
            for root, dirs, files in os.walk(self.entity_dirpath):
                for file in files:
                    filepath = os.path.join(root, file)
                    for line in open(filepath, 'r', encoding='utf-8'):
                        res = json.loads(line)
                        res_dict = {}
                        content=res["originalText"]
                        for i in enumerate(res["entities"]):
                            start = int(i[1]['start_pos'])
                            end= int(i[1]['end_pos'])
                            label=i[1]["label_type"]
                            label_id = self.label_dict.get(label)
                            for i in range(start, end+1):
                                if i == start:
                                    label_cate = label_id + '-B'
                                else:
                                    label_cate = label_id + '-I'
                                res_dict[i] = label_cate
                        for indx, char in enumerate(content):
                            char_label = res_dict.get(indx, 'O')
                            print(char, char_label)
                            f.write(char + '\t' + char_label + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
