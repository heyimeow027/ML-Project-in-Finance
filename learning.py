import os
import time
import dataProcess
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
t1 = time.time()
if "corr" not in os.listdir():
    os.mkdir("corr")
if "learningData" not in os.listdir():
    os.mkdir("learningData")
#数据预处理
dataProcess.keywordCorr()
dataProcess.searchIndex()
dataProcess.priceShift()
x1_data = []
x2_data = []
x_data = []
y_data = []
for i in os.listdir("learningData"):
    if "Shift.csv" in i:
        df = pd.read_csv(os.path.join("learningData", i))
        x1_data.append(df["lnPrice"].tolist())
        x2_data.append(df["searchIndex"].tolist())
        y_data.append(df["shiftPrice"].tolist())
for i in range(len(x1_data)):
    x_data.append([])
    x_data[i] = [[x1_data[i][j], x2_data[i][j]] for j in range(len(x1_data[i]))]
svr_linear= SVR(kernel="linear", C=32)
svr_rbf = SVR(kernel="rbf", C=32, gamma=0.125)
svr_sigmoid = SVR(kernel="sigmoid", C=32, gamma=0.125, coef0=0.1)
svr_sigmoid1 = SVR(kernel="sigmoid", C=32, gamma=0.25, coef0=0.03125)
GBR_ls1 = GBR(loss="squared_error", learning_rate=0.1, max_depth=3, n_estimators=20)
GBR_ls2 = GBR(loss="squared_error", learning_rate=0.45, max_depth=3, n_estimators=30)
GBR_lad1 = GBR(loss="absolute_error", learning_rate=0.05, max_depth=3, n_estimators=190)
GBR_lad2 = GBR(loss="absolute_error", learning_rate=0.4, max_depth=3, n_estimators=80)
GBR_huber1 = GBR(loss="huber", learning_rate=0.1, max_depth=3, n_estimators=20)
GBR_huber2 = GBR(loss="huber", learning_rate=0.1, max_depth=3, n_estimators=60)
Error = []
ErrorG = []
i = 0
for t in range(len(y_data)):
    x1_test, x_test, y_test = np.array(x1_data[t]).reshape(-1, 1), np.array(x_data[t]), np.array(y_data[t])
    x1_train, x_train, y_train = [], [], []
    for k in range(len(y_data)):
        if k != t:
            x1_train.extend(x1_data[k])
            x_train.extend(x_data[k])
            y_train.extend(y_data[k])
    x1_train = np.array(x1_train).reshape(-1, 1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    Error.append([])
    ErrorG.append([])
    svr_linear.fit(x1_train, y_train)
    y_linear = svr_linear.predict(x1_test)
    Error[i].append(mean_absolute_error(y_test, y_linear))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_linear)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_linear))
    svr_rbf.fit(x1_train, y_train)
    y_rbf = svr_rbf.predict(x1_test)
    Error[i].append(mean_absolute_error(y_test, y_rbf))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_rbf)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_rbf))
    svr_sigmoid.fit(x1_train, y_train)
    y_sigmoid = svr_sigmoid.predict(x1_test)
    Error[i].append(mean_absolute_error(y_test, y_sigmoid))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_sigmoid)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_sigmoid))
    svr_linear.fit(x_train, y_train)
    y_linear = svr_linear.predict(x_test)
    Error[i].append(mean_absolute_error(y_test, y_linear))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_linear)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_linear))
    svr_rbf.fit(x_train, y_train)
    y_rbf = svr_rbf.predict(x_test)
    Error[i].append(mean_absolute_error(y_test, y_rbf))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_rbf)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_rbf))
    svr_sigmoid1.fit(x_train, y_train)
    y_sigmoid = svr_sigmoid1.predict(x_test)
    Error[i].append(mean_absolute_error(y_test, y_sigmoid))
    Error[i].append(math.sqrt(mean_squared_error(y_test, y_sigmoid)))
    Error[i].append(mean_absolute_percentage_error(y_test, y_sigmoid))
    GBR_ls1.fit(x1_train, y_train)
    y_ls = GBR_ls1.predict(x1_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_ls))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_ls)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_ls))
    GBR_lad1.fit(x1_train, y_train)
    y_lad = GBR_lad1.predict(x1_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_lad))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_lad)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_lad))
    GBR_huber1.fit(x1_train, y_train)
    y_huber = GBR_huber1.predict(x1_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_huber))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_huber)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_ls))
    GBR_ls2.fit(x_train, y_train)
    y_ls = GBR_ls2.predict(x_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_ls))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_ls)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_ls))
    GBR_lad2.fit(x_train, y_train)
    y_lad = GBR_lad2.predict(x_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_lad))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_lad)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_lad))
    GBR_huber2.fit(x_train, y_train)
    y_huber = GBR_huber2.predict(x_test)
    ErrorG[i].append(mean_absolute_error(y_test, y_huber))
    ErrorG[i].append(math.sqrt(mean_squared_error(y_test, y_huber)))
    ErrorG[i].append(mean_absolute_percentage_error(y_test, y_ls))
    i += 1
Error1 = []
Error2 = []
for j in range(len(Error[0])):
    sum1 = 0
    sum2 = 0
    for k in range(len(Error)):
        sum1 += Error[k][j]
        sum2 += ErrorG[k][j]
    Error1.append(sum1/len(y_data))
    Error2.append(sum2/len(y_data))
data_single = [Error1[:3], Error1[3:6], Error1[6:9]]
df_single = pd.DataFrame(data_single, columns=["MAE", "RMSE", "MAPE"])
df_single.index = ["linear", "rbf", "sigmoid"]
df_single.to_csv("SVR_DoubleX.csv")
data_double = [Error1[9:12], Error1[12:15], Error1[15:]]
df_double = pd.DataFrame(data_double, columns=["MAE", "RMSE", "MAPE"])
df_double.index = ["linear", "rbf", "sigmoid"]
df_double.to_csv("SVR_SingleX.csv")
data_s = [Error2[:3], Error2[3:6], Error2[6:9]]
df_s = pd.DataFrame(data_s, columns=["MAE", "RMSE", "MAPE"], index=["ls", "lad", "huber"])
df_s.to_csv("GBR_DoubleX.csv")
data_d = [Error2[9:12], Error2[12:15], Error2[15:]]
df_d = pd.DataFrame(data_d, columns=["MAE", "RMSE", "MAPE"], index=["ls", "lad", "huber"])
df_d.to_csv("GBR_SingleX.csv")
t2 = time.time()
print(t2 - t1)
