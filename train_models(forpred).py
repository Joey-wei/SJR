# -*-coding = utf-8 -*-
# @Time : 2022/9/28 20:57
# @File : train_models(forpred).py
# @software : PyCharm
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler
import os
import warnings


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


# need to be trained, will be used to transfer
file_name = os.listdir('data(2016-2019)')
file_path = ['data(2016-2019)/' + file_name[i] for i in range(len(file_name))]
print(file_path)

for i in range(len(file_path)):
    try:
        df = pd.read_csv(file_path[i])
        df = df.dropna()     # drop nan
        # make training set
        X = np.matrix(df.iloc[:, 1:-1])
        Y = np.matrix(df.iloc[:, -1:])
        # train set
        X_train, y_train = X, Y

        X_standardScaler = StandardScaler()
        X_standardScaler.fit(X_train)
        X_train_standard = X_standardScaler.transform(X_train)

        y_standardScaler = StandardScaler()
        y_standardScaler.fit(y_train)
        y_train_standard = y_standardScaler.transform(y_train)

        #
        m_ann = ANN(file_name[i].split('.')[0])     # ann
        m_ann.fit(X_train_standard, y_train_standard)
        m_ann.save()
    except ValueError:
        continue
