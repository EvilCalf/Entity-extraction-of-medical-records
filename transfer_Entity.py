import os


class TransferData:
    def __init__(self):
        self.label_dict = {
            '检查和检验': 'CHECK',
            '症状和体征': 'SIGNS',
            '疾病和诊断': 'DISEASE',
            '治疗': 'TREATMENT',
            '身体部位': 'BODY',
            '药品': 'TRUGS'}

        self.cate_dict = {
            'O': 0,
            'TREATMENT-I': 1,
            'TREATMENT-B': 2,
            'BODY-B': 3,
            'BODY-I': 4,
            'SIGNS-I': 5,
            'SIGNS-B': 6,
            'CHECK-B': 7,
            'CHECK-I': 8,
            'DISEASE-I': 9,
            'DISEASE-B': 10,
            'TRUGS-B': 11,
            'TRUGS-I': 12
        }
        self.entity_dirpath = "Medical Entity"
        return

    def transfer(self):
        with open('data\\train.txt', encoding="utf-8", mode="a") as f:
            for root, dirs, files in os.walk(self.entity_dirpath):
                for file in files:
                    filepath = os.path.join(root, file)
                    res_dict = {}
                    for line in open(filepath, 'r', encoding='utf-8'):
                        res = line.strip().split('	')
                        label = res[1]
                        label_id = self.label_dict.get(label)
                        for i in range(0, len(res[0])):
                            if i == 0:
                                label_cate = label_id + '-B'
                            else:
                                label_cate = label_id + '-I'
                            res_dict[i] = label_cate
                        for indx, char in enumerate(res[0]):
                            char_label = res_dict.get(indx, 'O')
                            print(char, char_label)
                            f.write(char + '\t' + char_label + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
