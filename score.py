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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMNER:
    def __init__(self):
        keras.backend.clear_session()
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.test_path = os.path.join(cur, 'train/yidu_test.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(
            cur, 'model/token_vec_300.bin')  # 可自行修改预训练词向量
        self.model_path = os.path.join(
            cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.word_dict = self.load_worddict()
        self.class_dict = {
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
        self.label_dict = {j: i for i, j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 100
        self.BATCH_SIZE = 64
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 1000  # 预测的时候为输入段落的最长长度
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)
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
        x_test=[]
        txt=""
        for data in self.datas:
            txt=txt.join(str(i) for i in data[0])
            x_test.append(txt)
            txt=""
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

    def recall_score(self,y_pre):
        sample_y=[]
        y_true=[]
        for data in self.datas:
            for tag in data[1]:
                sample_y.append(tag)
            y_true.append(sample_y)
            sample_y=[]
        T=0
        Total=0
        for i,tags in enumerate(y_true):
            for j,tag in enumerate(tags):
                if tag=='O':
                    continue
                Total=Total+1
                if y_pre[i][j].split("_")[0]==tag.split("_")[0]:
                    T=T+1
        return T/Total

    def precision_score(self,y_pre):
        sample_y=[]
        y_true=[]
        for data in self.datas:
            for tag in data[1]:
                sample_y.append(tag)
            y_true.append(sample_y)
            sample_y=[]
        T=0
        Total=0
        for i,tags in enumerate(y_pre):
            for j,tag in enumerate(tags):
                if tag=='O':
                    continue
                Total=Total+1
                if y_true[i][j].split("_")[0]==tag.split("_")[0]:
                    T=T+1
        return T/Total

    def test_model(self):
        x_test= self.modify_data()
        predictions=[]
        for text in x_test:
            str = self.build_input(text)
            raw = self.model.predict(str)[0][-self.TIME_STAMPS:]
            result = [np.argmax(row) for row in raw]
            tags = [self.label_dict[i] for i in result][len(result)-len(text):]
            predictions.append(tags)
        # f1_cnt=f1_score(self.datas[1],predictions)
        precision_cnt=self.precision_score(predictions)
        recall_cnt=self.recall_score(predictions)
        # print(f1_cnt)
        print(precision_cnt)
        print(recall_cnt)
        print((2 * precision_cnt * recall_cnt)/(precision_cnt + recall_cnt))


if __name__ == '__main__':
    ner = LSTMNER()
    ner.test_model()
