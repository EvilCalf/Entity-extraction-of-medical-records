import csv
import json
import os

import keras
import numpy as np
from keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout, Embedding,
                          TimeDistributed)
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers.crf import CRF

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMNER:
    def __init__(self):
        keras.backend.clear_session()
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'train/web.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(
            cur, 'model/token_vec_300.bin')  # 可自行修改预训练词向量
        self.model_path = os.path.join(
            cur, 'model/model/tokenvec_bilstm2_crf_model_20.h5')
        self.word_dict = self.load_worddict()
        self.class_dict = {
            "O": 0,
            "B-DISEASE": 1,
            "I-DISEASE": 2,
            "B-SIGN": 3,
            "I-SIGN": 4,
            "B-MARGIN": 5,
            "I-MARGIN": 6,
            "B-DIAMETER": 7,
            "I-DIAMETER": 8,
            "B-TESTPROC": 9,
            "I-TESTPROC": 10,
            "B-TREATMENT": 11,
            "I-TREATMENT": 12,
            "B-ANATOMY": 13,
            "I-ANATOMY": 14,
            "B-NATURE": 15,
            "I-NATURE": 16,
            "B-SHAPE": 17,
            "I-SHAPE": 18,
            "B-DENSITY": 19,
            "I-DENSITY": 20,
            "B-BOUNDARY": 21,
            "I-BOUNDARY": 22,
            "B-LUNGFIELD": 23,
            "I-LUNGFIELD": 24,
            "B-TEXTURE": 25,
            "I-TEXTURE": 26,
            "B-TRANSPARENCY": 27,
            "I-TRANSPARENCY": 28
        }
        self.label_dict = {j: i for i, j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 100
        self.BATCH_SIZE = 64
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 1000  # 预测的时候为输入段落的最长长度
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)

    '加载词表'

    def load_worddict(self):
        vocabs = [line.strip()
                  for line in open(self.vocab_path, encoding='utf-8')]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    '''构造输入，转换成所需形式'''

    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = 'UNK'
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x

    def predict(self, text):
        y_pre=[]
        str = self.build_input(text)
        raw = self.model.predict(str)[0][-self.TIME_STAMPS:]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        tags = [self.label_dict[i] for i in result][len(result)-len(text):]
        res = list(zip(chars, tags))
        for i,tag in enumerate(tags):
            y_pre.append(tag)
        return res,y_pre

    '''使用预训练向量进行模型训练'''

    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        # embedding_layer = Embedding(self.VOCAB_SIZE + 1,
        #                             self.EMBEDDING_DIM,
        #                             weights=[self.embedding_matrix],
        #                             input_length=self.TIME_STAMPS,
        #                             trainable=False,
        #                             mask_zero=True)
        model.add(Embedding(self.VOCAB_SIZE+1, self.EMBEDDING_DIM,
                            mask_zero=True))  # Random embedding
        # model.add(embedding_layer)
        model.add(Bidirectional(GRU(256, return_sequences=True,
                                    kernel_regularizer=keras.regularizers.l2(0.01))))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, return_sequences=True,
                                    kernel_regularizer=keras.regularizers.l2(0.01))))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES,
                                        kernel_regularizer=keras.regularizers.l2(0.01))))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True,
                        kernel_regularizer=keras.regularizers.l2(0.01))
        model.add(crf_layer)
        Adam = keras.optimizers.adam(lr=0.005)
        model.compile(Adam, loss=crf_layer.loss_function,
                      metrics=[crf_layer.accuracy])
        model.summary()
        return model

    def output(self, cnt):
        output = []
        flag = 0
        start = []
        end = []
        tags = []
        for i, tag in enumerate(cnt):
            if tag == 'O':
                if flag == 1:
                    end = i-1
                    output.append([tags, start, end])
                flag = 0
                continue
            if tag.split("-")[0] == 'B':
                if flag == 1:
                    end = i-1
                flag = 1
                start = i
                tags = tag.split("-")[1]
                continue
        return output


if __name__ == '__main__':
    ner = LSTMNER()
    res = []
    ans_json = []
    for root, dirs, files in os.walk("data_out"):
        for file in files:
            with open(root+"/"+file, "r", encoding='utf-8') as f:
                reader = csv.reader(f)
                result = list(reader)
                total = len(result)
                for i, strs in enumerate(result):
                    ans,y_pre = ner.predict(strs[0])
                    res.append(ans)
                    print(str(i)+"/"+str(total-1))
                    y_pre = ner.output(y_pre)
                    filename = open("data_out_json/"+file.replace(".csv", "")+"_"+str(i)+"_P.txt", 'w+',encoding='utf-8')  
                    for value in y_pre:  
                        filename.write(str(value)+'\n') 
                    filename.close() 
                with open("data_out_json/"+file.replace(".csv", "")+".json", 'w', encoding='utf-8') as file_obj:
                    for i, strs in enumerate(res):
                        ans_json.append(
                            {'data': [{'chars': index[0], 'tags': index[1]} for index in res[i]]})
                    json.dump(ans_json, file_obj, indent=4, ensure_ascii=False)
                filename = open("data_out_json/"+file.replace(".csv", "")+"_BIO.txt", 'w+',encoding='utf-8')  
                for i, strs in enumerate(res):
                    for j,value in enumerate(strs):
                        filename.write(str(value[0])+'\t'+str(value[1])+'\n') 
                filename.close() 
                