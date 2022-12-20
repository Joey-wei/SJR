# -*-coding = utf-8 -*-
# @Time : 2022/9/28 21:21
# @File : transition.py
# @software : PyCharm
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import os

file_name = os.listdir('models(2016-2019)')
file_path = ['models(2016-2019)/' + file_name[i] for i in range(len(file_name))]
print(file_path)
# transfer data(2020-2021)
df = pd.read_csv('ATHENS.csv')
df = df.dropna()     # drop nan
# train set
X_train = np.matrix(df.iloc[:366, 1:-1])
y_train = np.matrix(df.iloc[:366, -1:])
# test set
X_test = np.matrix(df.iloc[366:, 1:-1])
y_test = np.matrix(df.iloc[366:, -1:])

X_standardScaler = StandardScaler()
X_standardScaler.fit(X_train)
X_train_standard = X_standardScaler.transform(X_train)
X_test_standard = X_standardScaler.transform(X_test)

y_standardScaler = StandardScaler()
y_standardScaler.fit(y_train)
y_train_standard = y_standardScaler.transform(y_train)


# no miss
for i in range(len(file_path)):
    try:
        # ANN model
        pred_arr = []
        for j in range(5):
            m_ann = load_model(file_path[i])
            m_ann.fit(X_train_standard, y_train_standard, epochs=14, batch_size=32)
            pred = m_ann.predict(X_test_standard)
            pred = y_standardScaler.inverse_transform(pred)
            # plt.plot(pred(drop0.5)1, c='r')
            # plt.plot(y_test, c='b')
            # plt.title(file_name[i])
            # plt.show()
            pred_arr.append(pred)

        pred = pred_arr[0]
        for k in range(1, 5):
            pred = np.add(pred, pred_arr[k])
        pred /= 5
        print(file_name[i].split(".")[0])
        pred = pd.DataFrame(pred)
        pred.to_csv(f'pred/{file_name[i].split(".")[0]}.csv')
        print(r2_score(y_test, pred))
        print(mean_absolute_percentage_error(y_test, pred))

    except Exception:
        continue

