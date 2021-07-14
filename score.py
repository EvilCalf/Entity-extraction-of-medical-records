from enum import EnumMeta
import os

import keras
import numpy as np
from keras import backend as K
from keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout, Embedding,
                          TimeDistributed)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers.crf import CRF
from builtins import str
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMNER:
    def __init__(self):
        keras.backend.clear_session()
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.test_path = os.path.join(cur, 'train/sict_train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(
            cur, 'model/token_vec_300.bin')  # 可自行修改预训练词向量
        self.model_path = os.path.join(
            cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
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
        self.datas = self.build_data()

    def load_worddict(self):
        vocabs = [line.strip()
                  for line in open(self.vocab_path, encoding='utf-8')]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    '''构造数据集'''

    def build_data(self):
        datas = []
        sample_x = []
        sample_y = []
        for line in open(self.test_path, 'r', encoding='utf-8'):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            if char in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        return datas

    '''将数据转换成keras所需的格式'''

    def modify_data(self):
        x_test = []
        txt = ""
        for data in self.datas:
            txt = txt.join(str(i) for i in data[0])
            x_test.append(txt)
            txt = ""
        return x_test

    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = 'UNK'
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x
    # '''加载预训练词向量'''

    # def load_pretrained_embedding(self):
    #     embeddings_dict = {}
    #     with open(self.embedding_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             values = line.strip().split(' ')
    #             if len(values) < 300:
    #                 continue
    #             word = values[0]
    #             coefs = np.asarray(values[1:], dtype='float32')
    #             embeddings_dict[word] = coefs
    #     print('Found %s word vectors.' % len(embeddings_dict))
    #     return embeddings_dict

    # '''加载词向量矩阵'''

    # def build_embedding_matrix(self):
    #     embedding_dict = self.load_pretrained_embedding()
    #     embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
    #     for word, i in self.word_dict.items():
    #         embedding_vector = embedding_dict.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector
    #     return embedding_matrix

    # '''使用预训练向量进行模型训练'''

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

    def test_model(self):
        f = open("score.txt", 'a+', encoding='utf-8')
        for root, dirs, files in os.walk("model/model/"):
            x_test = self.modify_data()
            epoch = 0
            for file in files:
                epoch = epoch+1
                self.model.load_weights(root+file)
                y_pre = []
                filename = open(self.test_path.replace(".txt","")+"_Result.txt", 'w+',encoding='utf-8')  
                for text in x_test:
                    string = self.build_input(text)
                    raw = self.model.predict(string)[0][-self.TIME_STAMPS:]
                    result = [np.argmax(row) for row in raw]
                    tags = [self.label_dict[i]
                            for i in result][len(result)-len(text):]
                    for i,tag in enumerate(tags):
                        y_pre.append(tag)
                        filename.write(text[i]+'\t'+str(tag)+'\n') 
                filename.close()
                y_true = []
                for data in self.datas:
                    for tag in data[1]:
                        y_true.append(tag)
                print("第"+str(epoch)+"轮模型结果:")
                y_pre = self.output(y_pre)
                y_true = self.output(y_true)
                filename = open(self.test_path.replace(".txt","")+"_P.txt", 'w+',encoding='utf-8')  
                for value in y_pre:  
                     filename.write(str(value)+'\n') 
                filename.close() 
                filename = open(self.test_path.replace(".txt","")+"_T.txt", 'w+',encoding='utf-8')  
                for value in y_true:  
                     filename.write(str(value)+'\n') 
                filename.close()  
                c = [x for x in y_pre if x not in y_true]
                d = [y for y in y_true if y not in y_pre]
                TP = len(y_pre)-len(c)
                FP = len(c)
                FN = len(d)
                precision_score = TP/(TP+FP)
                print("precision_score=TP/(TP+FP):"+str(precision_score))
                recall_score = TP/(TP+FN)
                print("recall_score=TP/(TP+FN):"+str(recall_score))
                f1 = precision_score*recall_score * \
                    2/(precision_score+recall_score)
                print(
                    "F1=precision_score*recall_score*2/(precision_score+recall_score):"+str(f1))
                f.write(str(TP)+'\t'+str(FP)+'\t'+str(FN)+'\t'+str(precision_score) + '\t' +str(recall_score) + '\t' + str(f1) + '\n')
        f.close()


if __name__ == '__main__':
    ner = LSTMNER()
    ner.test_model()
