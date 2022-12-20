# -*-coding = utf-8 -*-
# @Time : 2022/9/25 21:19
# @File : train_models(R2self).py
# @software : PyCharm
import joblib
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings("ignore")


# ANN model
class ANN:
    def __init__(self, name):
        self.name = name
        self.model = models.Sequential()
        self.model.add(layers.Dense(1024, activation='sigmoid', name="Dense_1"))
        self.model.add(layers.Dense(1024, activation='sigmoid', name="Dense_2"))
        self.model.add(layers.Dense(1024, activation='sigmoid', name="Dense_3"))
        self.model.add(layers.Dense(1, name="Dense_4"))
        self.model.compile(loss='mse', optimizer='Adam')

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, batch_size=64, epochs=50)

    def pred(self, x):
        y = self.model.predict(x)
        return y

    def save(self):
        self.model.save(f'models(2016-2019)/{self.name}.h5')


# file path
file_name = os.listdir('data(2016-2019)')
file_path = ['data(2016-2019)/' + file_name[i] for i in range(len(file_name))]

dic = dict()
res = []
for i in range(len(file_path)):
    try:
        df = pd.read_csv(file_path[i])
        # train set
        train = df.iloc[:-365, :]
        train = train.dropna()
        # test set
        test = df.iloc[-365:, :]
        test = test.dropna()

        X_train, X_test = np.matrix(train.iloc[:, 1:-1]), np.matrix(test.iloc[:, 1:-1])
        y_train, y_test = np.matrix(train.iloc[:, -1:]), np.matrix(test.iloc[:, -1:])
        #
        X_standardScaler = StandardScaler()
        X_standardScaler.fit(X_train)
        X_train_standard = X_standardScaler.transform(X_train)
        X_test_standard = X_standardScaler.transform(X_test)

        y_standardScaler = StandardScaler()
        y_standardScaler.fit(y_train)
        y_train_standard = y_standardScaler.transform(y_train)

        #
        m_ann = ANN(file_name[i].split('.')[0])     # ann
        m_ann.fit(X_train_standard, y_train_standard)

        pred = m_ann.pred(X_test_standard)
        pred = y_standardScaler.inverse_transform(pred)
        #
        dic[file_name[i]] = r2_score(y_test, pred)
        res.append(r2_score(y_test, pred))
        print(f'{file_name[i]}:{r2_score(y_test, pred)}')

    except ValueError:
        dic[file_name[i]] = -9999
        res.append(-9999)

print(dic)
print(res)
