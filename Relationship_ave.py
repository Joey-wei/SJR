import logging
import pandas as pd
import xlwt
import numpy as np
from minepy import MINE
from sklearn.preprocessing import MinMaxScaler
from upload_12_20.JSMIC import JSMIC


class RelationShip:

    def getData(self, data_file='dataset.xlsx', sheet_name='Sheet1'):
        # get original dataset
        df = pd.read_excel(data_file, sheet_name, header=0)  # Read the data, the format is dataFrame
        data = np.array(df.iloc[:, 1:])  # Get the pure data part, excluding the first column (number or time column)
        n, m = data.shape  # Get the number of rows and columns of data
        print('Data length:' + str(n) + '   Number:' + str(m))
        return data

    def getDataFrame(self, data_file='dataset.xlsx', sheet_name='Sheet1'):
        # get original dataset
        df = pd.read_excel(data_file, sheet_name, header=0)  # Read the data, the format is dataFrame
        return df

    def out_file(self, list_data,Doc_name,f_name):

        aa = np.array(list_data).reshape((-1,2))

        f = xlwt.Workbook()
        sheet_name = 'Sheet1'
        sh = f.add_sheet(sheet_name, cell_overwrite_ok=False)
        sh.write(0, 4, "a")
        sh.write(0, 5, "b")

        for l in range(len(aa)):
            sh.write(l + 1, 4, int(aa[l][0]))
            sh.write(l + 1, 5, int(aa[l][1]))


        f_xls = f_name + Doc_name + '.xls'
        f.save(f_xls)

    # correlation coefficient
    def correlation(self, x, y, method='pearson'):
        x = np.reshape(x,(-1,1))
        y = np.reshape(y,(-1,1))
        input = np.hstack((x, y))
        result = pd.DataFrame(input).corr(method=method)

        return np.round(result[0][1], 4)

    def split(self, x, length):
        num_s = int(len(x) / length)
        mat = np.zeros((num_s, length))
        for i in range(num_s):
            temp = x[i * length:i * length + length]
            mat[i, :] = np.array(temp).flatten()

        return mat

    def JS_MAT(self,x):
        m,n = np.array(x).shape
        js_mat = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                # JS = JSMIC().JSMIC_val(np.array(x[i]).flatten(), np.array(x[j]).flatten())
                JS = JSMIC().JS_div(np.array(x[i]).flatten(), np.array(x[j]).flatten())
                # mine = MINE(alpha=0.6, c=15)
                # mine.compute_score(np.array(x[i]).flatten(), np.array(x[j]).flatten())
                # JS = mine.mic()
                js_mat[i][j] = JS

        return js_mat

    def sort_index(self, arr):
        sort = np.argsort(arr)
        sort_R = np.argsort(sort) + 1

        return sort_R


if __name__ == '__main__':
    # Set the output format
    logger = logging.getLogger()
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    # Initialize Get data Modify data date
    rs = RelationShip()
    dataFrame = rs.getDataFrame('greece2012-2022-NS.xlsx','DT')
    dataFrame = dataFrame.fillna(0) # Fill the empty value with 0
    select_cols = dataFrame.columns[0]
    col_names = dataFrame.columns[:]
    dataFrame[select_cols] = pd.to_datetime(dataFrame[select_cols], format='%Y/%m/%d')
    # Remove the 29th to ensure data alignment
    dataFrame = dataFrame.drop(dataFrame.index[(dataFrame[select_cols] == '2012-02-29')])
    dataFrame = dataFrame.drop(dataFrame.index[(dataFrame[select_cols] == '2016-02-29')])
    dataFrame = dataFrame.drop(dataFrame.index[(dataFrame[select_cols] == '2020-02-29')])
    # Get year month day label
    ind_year = np.array(dataFrame[select_cols].dt.year)

    a = np.array(dataFrame['DT']).reshape((-1, 1))
    a_sel = np.where(ind_year == 2020)
    a = a[a_sel] # a is the target domain
    m = len(a)

    # A few percent before the selection is made into missing data, assigned a value of 0
    miss_percent = 0.5
    a_sort_index = np.argsort(np.array(a).flatten())
    miss_num = int(miss_percent*m)
    miss_index = a_sort_index[:miss_num]
    a_drop = np.delete(a, miss_index)


    b_mat = np.array(dataFrame.iloc[:, 2:])
    _, n = np.array(b_mat).shape


    a_normal = scaler.fit_transform(np.array(a_drop).reshape((-1, 1)))


    years = 4
    output = np.zeros((years, n))
    JS_MAT = np.zeros(n)
    # JS_MAT = np.zeros((years, n))
    SP_MAT = np.zeros((years, n))
    PE_MAT = np.zeros((years, n))
    MIC_MAT = np.zeros((years, n))

    SJR_year = 2019
    SJR_b_sel = np.where(ind_year == (SJR_year))
    SJR_b_mat_sel = b_mat[SJR_b_sel, :]
    SJR_b_mat_sel = np.reshape(SJR_b_mat_sel, (m, -1))

    for year in range(years):

        b_sel = np.where(ind_year == (2019-year))
        b_mat_sel = b_mat[b_sel, :]  # b is the source domain
        b_mat_sel = np.reshape(b_mat_sel, (m, -1))

        for i in range(n):
            b = np.array(b_mat_sel[:, i]).flatten()
            b_drop = np.delete(b, miss_index)
            b_normal = scaler.fit_transform(np.array(b_drop).reshape((-1, 1)))

            # SJR_b = np.array(SJR_b_mat_sel[:, i]).flatten()
            # SJR_b_normal = scaler.fit_transform(np.array(SJR_b).reshape((-1, 1)))

            # JS = JSMIC().JS_div((a_normal).flatten(), np.array(b_normal).flatten())

            spearman = rs.correlation(a_normal, b_normal, 'spearman')
            pearson = rs.correlation(a_normal, b_normal, 'pearson')
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(np.array(a_normal).flatten(), np.array(b_normal).flatten())
            mic = mine.mic()

            # JS_MAT[year][i] = JS
            SP_MAT[year][i] = np.abs(spearman)
            PE_MAT[year][i] = np.abs(pearson)
            MIC_MAT[year][i] = mic

    for i in range(n):
        x = np.array(b_mat[:, i]).flatten()
        mat = rs.split(x, m)
        simi = rs.JS_MAT(mat)
        simi_arr = np.unique(simi)
        simi_arr = np.delete(simi_arr, np.where(simi_arr == 1))
        simi_ave = np.average(simi_arr)
        # simi_arr = np.array(simi[0,1:]).flatten()
        # simi_ave = np.average(simi_arr)

        JS_MAT[i] = simi_ave

    JS_AVE = JS_MAT

    # JS_AVE = np.sum(JS_MAT, axis=0) / years
    SP_AVE = np.sum(SP_MAT, axis=0) / years

    SP_SORT_R = rs.sort_index(SP_AVE)
    JS_SORT_R = rs.sort_index(JS_AVE)
    SJR = (SP_SORT_R+JS_SORT_R)/(2*len(SP_SORT_R)) # SJR outputs


    PE_AVE = np.sum(PE_MAT, axis=0) / years
    MIC_AVE = np.sum(MIC_MAT, axis=0) / years

    workBook = xlwt.Workbook()
    sheet1 = workBook.add_sheet('SHEET1')
    sheet1.write(0, 1, "SJR")
    sheet1.write(0, 2, "JS")
    sheet1.write(0, 3, "SP")
    sheet1.write(0, 4, "PCC")
    sheet1.write(0, 5, "MIC")
    for j in range(n):
        sheet1.write(j + 1, 0, col_names[j + 2])
        # sheet1.write(j + 1, 1, SJR[j])
        # sheet1.write(j + 1, 2, JS_AVE[j])
        # sheet1.write(j + 1, 3, SP_AVE[j])
        # sheet1.write(j + 1, 4, PE_AVE[j])
        # sheet1.write(j + 1, 5, MIC_AVE[j])

        SJR_R = 15-rs.sort_index(SJR)
        JS_AVE_R = 15-rs.sort_index(JS_AVE)
        SP_AVE_R = 15-rs.sort_index(SP_AVE)
        PE_AVE_R = 15-rs.sort_index(PE_AVE)
        MIC_AVE_R = 15-rs.sort_index(MIC_AVE)

        sheet1.write(j + 1, 1, int(SJR_R[j]))
        sheet1.write(j + 1, 2, int(JS_AVE_R[j]))
        sheet1.write(j + 1, 3, int(SP_AVE_R[j]))
        sheet1.write(j + 1, 4, int(PE_AVE_R[j]))
        sheet1.write(j + 1, 5, int(MIC_AVE_R[j]))


    workBook.save("outputs/2020DT-miss50%_R_SJR.xls")

