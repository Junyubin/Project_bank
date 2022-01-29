import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K 
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import sys
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
set_session(sess)

WORKING_DIRECTORY = sys.path[0]
# WORKING_DIRECTORY = '/home/ctilab/dti_v3/test2/dti'


class AttackClassification:
    """
공격 분류를 위한 인공지능 학습 모델.
    """
    def __init__(self, version=None, mode=None, config=None):
        """
        :param version: 모델의 버전을 정의(information)
        :param mode: 현재까지는 'train' 만 제공, 향후 predict 기능 추가 예정
        TODO: predict mode 추가
        :param config: dictionary 형태로 설정값 전달 (common, train, predict, x_data_shape, y_data_shape)
        """
        self.version = version  # version 유효성 검사 필요??
        self.mode = mode
        self.name = config.get('common').get('model_name')
        self.x_data_shape = config.get('x_data_shape')
        self.y_data_shape = config.get('y_data_shape')
        self.att_name = config.get('att_name')

        if mode == 'train':
            self.lr = config.get("learning_rate")
            self.bs = config.get("batch_size")
            self.epochs = config.get("epochs")
            
            # optimizer 는 Adam 과 SGD 만 제공
            if config.get('optimizer') == 'Adam':
                self.optimizer = Adam(lr=self.lr)
            elif config.get('optimizer') == 'SGD':
                self.optimizer = SGD(lr=self.lr)
            else:
                # optimizer 값이 잘못된 경우
                # 1. Default optimizer 를 지정하여 적용
                self.optimizer = None
                # 2. 에러 발생
                print('optimizer 의 값이 잘못되었습니다. [Adam, SGD]')
        elif mode == 'predict':
            self.bs = config.get("batch_size")
            self.optimizer = config.get("optimizer")
            
        else:   # mode 값이 잘못된 경우 처리
            print('mode must be train or predict')

        self.save_path_model = WORKING_DIRECTORY + '/{}/{}'.format(config.get("common").get("model_path"), self.version)
        if not os.path.exists(self.save_path_model):
            os.makedirs(self.save_path_model)
            
        self.__set_neural_network()     # TODO: mode 에 대한 if 문 내에서 처리할 수 있??

    def __create_neural_network(self):
        """
        CNN 신경망을 생성하는 함수
        :return: CNN Network
        """
        cnn_model = Sequential([
            Input(shape=(self.x_data_shape[1:])),
            Conv1D(8, 10, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2, padding='same'),
            Conv1D(16, 10, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2, padding='same'),
            Conv1D(32, 10, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2, padding='same'),
            Conv1D(64, 10, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2, padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.y_data_shape[1], activation='softmax', name='predictions')
        ])
        cnn_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                          metrics=['categorical_crossentropy', 'accuracy'])
        return cnn_model

    def optimize_nn(self, X=None, Y=None, save=True):
        """
        지도 학습을 진행
        :param X: 입력 Data
        :param Y: Label
        :param save: 저장 여부
        """
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.001)
        ai_history = self.neural_network.fit(X, Y, epochs=self.epochs, batch_size=self.bs, shuffle=True, verbose=1,
                                             callbacks=[early_stop])
        if save:
            self.save_model()
            return 'MODEL HAS BEEN SAVED TO {}'.format(self.save_path_model), ai_history
        return ai_history

    def save_model(self):
        self.neural_network.save(self.save_path_model+'/{}.h5'.format(self.att_name))

    def __set_neural_network(self):
        """
        학습을 위한 신경망을 설정한다. mode 가 train 인 경우 새로운 신경망을 생성하고, predict 인 경우 학습된 모델을 불러온다.
        """
        if self.mode == 'train':
            K.clear_session()
            self.neural_network = self.__create_neural_network()
        else:
            self.neural_network = load_model(self.save_path_model+'/{}.h5'.format(self.att_name), custom_objects={'optimizer': self.optimizer})
        self.neural_network.summary()

    def predict(self, X=None):
        return self.neural_network.predict(X, batch_size=self.bs)

    def validation(self, X=None, Y=None):
        pred = self.predict(X).argmax(axis=1)
        true = Y.argmax(axis=1)
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average='weighted')
        recall = recall_score(true, pred, average='weighted')
        f1 = f1_score(true, pred, average='weighted')
        print("CONFUSION MATRIX")
        print(confusion_matrix(true, pred))
        print("ACCURACY SCORE : {}".format(accuracy))
        print("F1 SCORE : {}".format(f1))
        print("PRECISION SCORE : {}".format(precision))
        print("RECALL SCORE : {}".format(recall))

class DecisionTreeClassification:
    """
    Decision Tree 분류를 위한 인공지능 학습 모델.
    """
    def __init__(self, version=None, mode=None, config=None):
        """
        :param version: 모델의 버전을 정의(information)
        :param mode: 현재까지는 'train' 만 제공, 향후 predict 기능 추가 예정
        TODO: predict mode 추가
        :param config: dictionary 형태로 설정값 전달 (common, train, predict, x_data_shape, y_data_shape)
        """
        self.version = version  # version 유효성 검사 필요??
        self.mode = mode
        
        if mode == 'train':
            self.max_depth = config.get("max_depth")
        
        self.save_path_model = WORKING_DIRECTORY + '/{}/{}'.format(config.get("common").get("model_path"), self.version)
        self.__set_decision_tree()

        if not os.path.exists(self.save_path_model):
            os.makedirs(self.save_path_model)
        
    def __create_decision_tree(self):
        dt_model = DecisionTreeClassifier(max_depth = self.max_depth)
        return dt_model
    
    def fit_decision_tree(self, X=None, Y=None, save=True):
        """
        지도 학습을 진행
        :param X: 입력 Data
        :param Y: Label
        :param save: 저장 여부
        """
        ai_history = self.decision_tree.fit(X, Y)
        if save:
            self.save_model()
        return 'MODEL HAS BEEN SAVED TO {}'.format(self.save_path_model), ai_history
        return None, ai_history     

    def save_model(self):
        with open('{}/dt_model.pickle'.format(self.save_path_model), "wb") as fw:
            pickle.dump(self.decision_tree, fw)
        
    def load_model(self):
        print('{}/dt_model.pickle'.format(self.save_path_model))
        with open('{}/dt_model.pickle'.format(self.save_path_model), "rb") as fr:
            dt_model = pickle.load(fr)
        return dt_model

    def __set_decision_tree(self):
        """
        학습을 위한 신경망을 설정한다. mode 가 train 인 경우 새로운 신경망을 생성하고, predict 인 경우 학습된 모델을 불러온다.
        """
        if self.mode == 'train':
            self.decision_tree = self.__create_decision_tree()
        else:
            self.decision_tree = self.load_model()

    def predict(self, X=None):
        return self.decision_tree.predict(X)

    def validation(self, X=None, Y=None):        
        Y = Y.values
        pred = self.predict(X).argmax(axis=1)
        true = Y.argmax(axis=1)
        print("CONFUSION MATRIX")
        print(confusion_matrix(true, pred))
        print("ACCURACY SCORE : {}".format(accuracy_score(true, pred)))        
        return true, pred           
