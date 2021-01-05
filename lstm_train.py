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
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'train/yidu_train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(
            cur, 'model/token_vec_300.bin')  # 可自行修改预训练词向量
        self.model_path = os.path.join(
            cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.datas, self.word_dict = self.build_data()
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
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 1000
        self.BATCH_SIZE = 64
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 250  # 最长单句长度
        self.embedding_matrix = self.build_embedding_matrix()

    '''构造数据集'''

    def build_data(self):
        datas = []
        sample_x = []
        sample_y = []
        vocabs = {'UNK'}
        for line in open(self.train_path, 'r', encoding='utf-8'):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)
            if char in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    '''将数据转换成keras所需的格式'''

    def modify_data(self):
        x_train = [[self.word_dict[char] for char in data[0]]
                   for data in self.datas]
        y_train = [[self.class_dict[label] for label in data[1]]
                   for data in self.datas]
        x_train = pad_sequences(x_train, self.TIME_STAMPS)
        y = pad_sequences(y_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y, 2)
        return x_train, y_train

    '''保存字典文件'''

    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(wordlist))

    '''加载预训练词向量'''

    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''

    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

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
        model.add(Bidirectional(GRU(128, return_sequences=True,
                                    kernel_regularizer=keras.regularizers.l2(0.01))))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(64, return_sequences=True,
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

    '''训练模型'''

    def train_model(self):
        x_train, y_train = self.modify_data()
        model = self.tokenvec_bilstm2_crf_model()
        callbacks_list = [
            keras.callbacks.History(),
            keras.callbacks.ModelCheckpoint(self.model_path, monitor='crf_viterbi_accuracy', verbose=1,
                                            save_best_only=True, save_weights_only=True, mode='auto', period=1),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                              verbose=1, mode='auto', min_lr=0.00001),
            keras.callbacks.EarlyStopping(
                monitor='crf_viterbi_accuracy', min_delta=0.001, patience=6, verbose=0, mode='auto')
        ]
        if os.path.exists(self.model_path):
            model.load_weights(self.model_path)
        history = model.fit(x_train[:], y_train[:], validation_split=0.2,
                            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, callbacks=callbacks_list)
        return model


if __name__ == '__main__':
    ner = LSTMNER()
    ner.train_model()
