import os
from collections import Counter


class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {
            '影像检查': 'TESTPROC',
            '实验室检验': 'TESTPROC',
            '疾病和诊断': 'DISEASE',
            '手术': 'TREATMENT',
            '解剖部位': 'SIGN',
            '药物': 'DRUGS',
            '检查和检验': 'TESTPROC',
            '症状和体征': 'SYMPTOM',
            '治疗': 'TREATMENT',
            '身体部位': 'SIGN'
        }

        self.cate_dict = {
            'O': 0,  # 非实体
            'DISEASE-B': 1,  # 疾病或者综合征
            'DISEASE-I': 2,
            'INJURY-B': 3,  # 外伤
            'INJURY-I': 4,
            'VIRUS-B': 5,  # 病毒细菌
            'VIRUS-I': 6,
            'TYPE-B': 7,  # 疾病分类/等级
            "TYPE-I": 8,
            'SYMPTOM-B': 9,  # 自诉症状
            'SYMPTOM-I': 10,
            'SIGN-B': 11,  # 异常检查结果/体征
            'SIGN-I': 12,
            'TESTPROC-B': 13,  # 检查过程
            'TESTPROC-I': 14,
            'TESTITEM-B': 15,  # 检查项
            'TESTITEM-I': 16,
            'TESTEQUIP-B': 17,  # 检查设备
            'TESTEQUIP-I': 18,
            'TREATMENT-B': 19,  # 治疗过程
            'TREATMENT-I': 20,
            'DRUGS-B': 21,  # 药物
            'DRUGS-I': 22,
            'DOSE-B': 23,  # 剂量
            'DOSE-I': 24
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
