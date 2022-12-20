# -*-coding = utf-8 -*-
# @Time : 2022/10/4 19:21
# @File : transition(drop).py
# @software : PyCharm
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from keras.models import load_model

warnings.filterwarnings("ignore")

file_name = os.listdir('models(2016-2019)')
file_path = ['models(2016-2019)/' + file_name[i] for i in range(len(file_name))]

# transfer data(2020-2021)
file = 'ATHENS.csv'
data = pd.read_csv(file)
data_2020 = data.iloc[:366, :]
data_2021 = data.iloc[366:, :]

data_2020 = data_2020.sort_values(by=["ATHENS"], ascending=[True])
data_2020 = data_2020.dropna()     # drop nan

# miss rate
drop = [0.1, 0.3, 0.5, 0.7, 0.9]

for a in range(len(drop)):
    df = data_2020.iloc[int(drop[a]*len(data_2020)):, :]

    # train set
    X_train = np.matrix(df.iloc[:, 1:-1])
    y_train = np.matrix(df.iloc[:, -1:])

    # test set
    X_test = np.matrix(data_2021.iloc[:, 1:-1])
    y_test = np.matrix(data_2021.iloc[:, -1:])

    X_standardScaler = StandardScaler()
    X_standardScaler.fit(X_train)
    X_train_standard = X_standardScaler.transform(X_train)
    X_test_standard = X_standardScaler.transform(X_test)

    y_standardScaler = StandardScaler()
    y_standardScaler.fit(y_train)
    y_train_standard = y_standardScaler.transform(y_train)
    arr1 = []
    arr2 = []
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
            print(r2_score(y_test, pred))
            arr1.append(r2_score(y_test, pred))
            print(mean_absolute_percentage_error(y_test, pred))
            arr2.append(mean_absolute_percentage_error(y_test, pred))
            pred = pd.DataFrame(pred)
            pred.to_csv(f'pred(drop{drop[a]})3/{file_name[i].split(".")[0]}.csv')
            print(f'pred(drop{drop[a]})3/{file_name[i].split(".")[0]}.csv')

        except Exception:
            continue
    print(arr1)
    print(arr2)

