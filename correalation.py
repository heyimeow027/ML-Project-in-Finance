import os

import pandas as pd
import numpy as np
import math
#以交易量为权重对样本股票进行加权平均合成行业日均价序列
def lnPrice(fname):
    df = pd.read_csv(fname, names=["stock", "date", "price", "tradeCnt"], header=0)
    dateList = df["date"].tolist()
    dateList = list(set(dateList))
    dateList.sort(reverse=False)
    priceList = []
    lnPriceList = []
    for i in dateList:
        dailyPrice = 0
        dailyTrade = 0
        for j in range(len(df)):
            if df["date"][j] == i:
                dailyPrice += df["price"][j] * df["tradeCnt"][j]
                dailyTrade += df["tradeCnt"][j]
        priceList.append(dailyPrice/dailyTrade)
        lnPriceList.append(math.log(dailyPrice/dailyTrade))
    priceData = {"Date":dateList, "Price":priceList, "lnPrice":lnPriceList}
    dfp = pd.DataFrame(priceData)
    return dfp
#筛选交易日关键词搜索量加入序列
def searchCnt(dfp, fnames):
    dfs = pd.read_excel(fnames, sheet_name=0, names=["keyword", "area", "date", "search", "pc", "mobile"], header=0)
    searchList = []
    dateList = dfp["Date"].tolist()
    for i in dateList:
        for j in range(len(dfs)):
            if dfs["date"][j] == i:
                searchList.append(dfs["search"][j])
    dfp["searchCnt"] = searchList
    return dfp
#得到K阶Pearson相关系数的最大值及对应阶数
def correlation(dfp, maxLag):
    corrdata = []
    for i in range(0, maxLag+1):
        xList = dfp["searchCnt"][:len(dfp)-i]
        yList = dfp["Price"][i:]
        x = np.array(xList)
        y = np.array(yList)
        rho = np.corrcoef(x, y)
        corrdata.append([rho[1][0], i])
    corrdata.sort(key=lambda x:math.fabs(x[0]), reverse=True)
    return corrdata[0]
