import os

import keras
import numpy as np
from keras import backend as K
from keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout, Embedding,
                          TimeDistributed)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers.crf import CRF
from keras.callbacks import Callback

import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'train/elf_train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(
            cur, 'model/token_vec_300.bin')  # 可自行修改预训练词向量
        self.model_path = os.path.join(
            cur, 'model/model/tokenvec_bilstm2_crf_model_20.h5')
        self.datas, self.word_dict = self.build_data()
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
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 500
        self.BATCH_SIZE = 32
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 250  # 最长单句长度
        # self.embedding_matrix = self.build_embedding_matrix()

    '''构造数据集'''

    def load_worddict(self):
        vocabs = [line.strip()
                  for line in open(self.vocab_path, encoding='utf-8')]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

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
        Adam = keras.optimizers.adamax(lr=0.005)
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
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                              verbose=1, mode='auto', min_lr=1e-9),
            keras.callbacks.ModelCheckpoint("model/model/tokenvec_bilstm2_crf_model_20.h5", monitor='crf_viterbi_accuracy',
                                            verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            # WeightsSaver(1)
        ]
        if os.path.exists(self.model_path):
            model.load_weights(self.model_path)
        history = model.fit(x_train[:], y_train[:], batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS, callbacks=callbacks_list)
        with open('trainHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        return model

# class WeightsSaver(Callback):
#     def __init__(self, N):
#         self.N = N
#         self.epoch = 0

#     def on_epoch_end (self, epoch, logs={}):
#         if self.epoch % self.N == 0:
#             name = 'model/model/model%08d.h5' % self.epoch
#             self.model.save_weights(name)
#         self.epoch += 1


if __name__ == '__main__':
    ner = LSTMNER()
    ner.train_model()
