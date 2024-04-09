import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
plt.rcParams["font.sans-serif"] = ["SimHei"]  #正常显示中文
plt.rcParams["axes.unicode_minus"] = False  #正常显示负号
def test(fname):
    print(fname.split(".")[0])
    df = copy.deepcopy(pd.read_csv(fname, names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期"))
    df.drop(columns=["Price", "searchIndex"], inplace=True)
    df.plot()
    plt.show()
    print("综合股价序列的ADF检验结果为：", ADF(df["综合股价"]))  #ADF检验
    plot_acf(df).show()  #自相关函数图
    plot_pacf(df, method="ywm").show()  #偏相关函数图
    #自相关图长期大于0，作一阶差分
    diffData = df.diff().dropna()
    diffData.columns=["综合股价差分"]
    diffData.plot()
    plt.show()
    plot_acf(diffData).show()
    plot_pacf(diffData, method="ywm").show()
    print("差分序列的ADF检验结果为：", ADF(diffData["综合股价差分"]))
df_cde= pd.read_csv("cde.csv", names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期")
df_cde.drop(columns=["Price", "searchIndex"], inplace=True)
price_cde = df_cde["综合股价"].tolist()
df_cs = pd.read_csv("cs.csv", names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期")
df_cs.drop(columns=["Price", "searchIndex"], inplace=True)
price_cs = df_cs["综合股价"].tolist()
df_yb = pd.read_csv("yb.csv", names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期")
df_yb.drop(columns=["Price", "searchIndex"], inplace=True)
price_yb = df_yb["综合股价"].tolist()
df_ym = pd.read_csv("ym.csv", names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期")
df_ym.drop(columns=["Price", "searchIndex"], inplace=True)
price_ym = df_ym["综合股价"].tolist()
df_ymkd = pd.read_csv("ymkd.csv", names=["日期", "Price", "综合股价", "searchIndex"], header=0, index_col="日期")
df_ymkd.drop(columns=["Price", "searchIndex"], inplace=True)
price_ymkd = df_ymkd["综合股价"].tolist()
#模型训练
model_cde = ARIMA(df_cde.head(100).values, order=[1,1,1]).fit()
print("cde模型报告为：\n", model_cde.summary())
model_cs = ARIMA(df_cs.head(100).values, order=[1,1,1]).fit()
print("cs模型报告为：\n", model_cs.summary())
model_yb = ARIMA(df_yb.head(100).values, order=[1,1,1]).fit()
print("yb模型报告为：\n", model_yb.summary())
model_ym = ARIMA(df_ym.head(100).values, order=[1,1,1]).fit()
print("ym模型报告为：\n", model_ym.summary())
model_ymkd = ARIMA(df_ymkd.head(100).values, order=[1,1,1]).fit()
print("ymkd模型报告为：\n", model_ymkd.summary())
Error = [[], [], []]
Error[0].append(mean_absolute_error(price_cde[100:], model_cde.forecast(35)))
Error[0].append(mean_absolute_error(price_cs[100:], model_cs.forecast(35)))
Error[0].append(mean_absolute_error(price_yb[100:], model_yb.forecast(35)))
Error[0].append(mean_absolute_error(price_ym[100:], model_ym.forecast(35)))
Error[0].append(mean_absolute_error(price_ymkd[100:], model_ymkd.forecast(35)))
Error[1].append(math.sqrt(mean_squared_error(price_cde[100:], model_cde.forecast(35))))
Error[1].append(math.sqrt(mean_squared_error(price_cs[100:], model_cs.forecast(35))))
Error[1].append(math.sqrt(mean_squared_error(price_yb[100:], model_yb.forecast(35))))
Error[1].append(math.sqrt(mean_squared_error(price_ym[100:], model_ym.forecast(35))))
Error[1].append(math.sqrt(mean_squared_error(price_ymkd[100:], model_ymkd.forecast(35))))
Error[2].append(mean_absolute_percentage_error(price_cde[100:], model_cde.forecast(35)))
Error[2].append(mean_absolute_percentage_error(price_cs[100:], model_cs.forecast(35)))
Error[2].append(mean_absolute_percentage_error(price_yb[100:], model_yb.forecast(35)))
Error[2].append(mean_absolute_percentage_error(price_ym[100:], model_ym.forecast(35)))
Error[2].append(mean_absolute_percentage_error(price_ymkd[100:], model_ymkd.forecast(35)))
for i in range(len(Error)):
    Error[i] = sum(Error[i])/len(Error[i])
'''plt.plot(price_cde[1:])
plt.plot(model_cde.predict(start=1))
plt.legend(("true", "predict"))
plt.show()
plt.plot(price_cs[1:])
plt.plot(model_cs.predict(start=1))
plt.legend(("true", "predict"))
plt.show()
plt.plot(price_yb[1:])
plt.plot(model_yb.predict(start=1))
plt.legend(("true", "predict"))
plt.show()
plt.plot(price_ym[1:])
plt.plot(model_ym.predict(start=1))
plt.legend(("true", "predict"))
plt.show()
plt.plot(price_ymkd[1:])
plt.plot(model_ymkd.predict(start=1))
plt.legend(("true", "predict"))
plt.show()'''
print(Error)
#test("cde.csv")
#test("cs.csv")
#test("yb.csv")
#test("ym.csv")
#test("ymkd.csv")