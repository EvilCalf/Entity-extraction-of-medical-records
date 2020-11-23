import os
from collections import Counter


class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {
            '影像检查': 'TEST',
            '实验室检验': 'TEST',
            '疾病和诊断': 'DISEASE',
            '手术': 'TREATMENT',
            '解剖部位': 'BODY',
            '药物': 'DRUGS',
            '检查和检验': 'TEST',
            '症状和体征': 'SIGN',
            '治疗': 'TREATMENT',
            '身体部位': 'BODY'
            }

        self.cate_dict = {
            'O': 0,
            'TREATMENT-B': 1,
            'TREATMENT-I': 2,
            'BODY-B': 3,
            'BODY-I': 4,
            'TEST-B': 5,
            'TEST-I': 6,
            'DISEASE-B': 7,
            'DISEASE-I': 8,
            'SIGN-B': 9,
            'SIGN-I': 10,
        }
        self.origin_path = os.path.join(cur, 'data_origin')
        self.train_filepath = os.path.join(cur, 'train/data_origin.txt')
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        count = 0
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                if 'original' not in filepath:
                    continue
                label_filepath = filepath.replace('.txtoriginal', '')
                print(filepath, '\t\t', label_filepath)
                content = open(filepath, 'r', encoding='utf-8').read().strip()
                res_dict = {}
                for line in open(label_filepath, 'r', encoding='utf-8'):
                    res = line.strip().split('	')
                    start = int(res[1])
                    end = int(res[2])
                    label = res[3]
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
