import os

import numpy as np
import pandas as pd
import correalation
import math
import copy
#遍历得到每个关键词与对应股价序列的最大相关系数
def keywordCorr():
    for i in os.listdir():
        if os.path.isdir(i) and "关键词" in os.listdir(i):
            corrData = []
            keywordName = []
            for j in os.listdir(i):
                if os.path.isfile(os.path.join(i, j)):
                    dfp = correalation.lnPrice(os.path.join(i, j))
                    filename = j.split("_")[1].split(".")[0] +"Corr.csv"
                    path = os.path.join(i, "关键词")
                    for k in os.listdir(path):
                        df = correalation.searchCnt(dfp, os.path.join(path, k))
                        corrData.append(correalation.correlation(df, 8))
                        keywordName.append(k.split("_")[0])
                    corrData.sort(key=lambda x:math.fabs(x[0]), reverse=True)
                    dfc = pd.DataFrame(corrData, columns=["Corrcoef", "Lag"], index=keywordName)
                    dfc.to_csv(os.path.join("corr", filename))
class weightValue:
    def __init__(self):
        self.keyword = []
        self.corr = []
        self.weight = []
    def addKeyword(self, e):
        self.keyword.append(e)
    def addCorr(self, e):
        self.corr.append(math.fabs(e))
    def isInKeyword(self, e):
        return e in self.keyword
    def setWeight(self):
        s = sum(self.corr)
        l = len(self.corr)
        self.weight = [None] * l
        for i in range(len(self.corr)):
            self.weight[i] = self.corr[i]/s
    def getWeight(self,e):
        assert e in self.keyword
        for i in range(len(self.keyword)):
            if e == self.keyword[i]:
                return self.weight[i]
#以相关系数绝对值为权重合成搜索量指数
def searchIndex():
    for i in os.listdir("corr"):
        name = i.split("C")[0]
        value = weightValue()
        df = pd.read_csv(os.path.join("corr", i), names=["Name", "Corrcoef", "Lag"], header=0)
        for j in range(len(df)):
            if math.fabs(df["Corrcoef"][j]) > 0.39:
                value.addCorr(df["Corrcoef"][j])
                value.addKeyword(df["Name"][j])
        value.setWeight()
        searchData = []
        for j in os.listdir():
            if os.path.isdir(j) and "price_"+name+".csv" in os.listdir(j):
                dfp = correalation.lnPrice(os.path.join(j, "price_"+name+".csv"))
                df = copy.deepcopy(dfp)
                for k in os.listdir(os.path.join(j, "关键词")):
                    if value.isInKeyword(k.split("_")[0]):
                        weight = value.getWeight(k.split("_")[0])
                        dfs = correalation.searchCnt(dfp, os.path.join(j, "关键词", k))
                        searchList = [item * weight for item in dfs["searchCnt"]]
                        searchData.append(searchList)
                search = []
                for l in range(len(searchData[0])):
                    s = 0
                    for t in range(len(searchData)):
                        s += searchData[t][l]
                    search.append(math.log(s))
                df["searchIndex"] = search
                df.to_csv(os.path.join("learningData", name+".csv"), index=False)
def priceShift():
    for i in os.listdir("learningData"):
        if "Shift" not in i:
            name = i.split(".")[0]
            df = copy.deepcopy(pd.read_csv(os.path.join("learningData", i)))
            df["shiftPrice"] = df["lnPrice"].shift(-1)
            df = df.drop(labels=len(df)-1, axis=0)
            df = df.drop(columns=["Price"])
            df.to_csv(os.path.join("learningData", name+"Shift.csv"), index=False)

x = (8*0.1337 + 7*0.1736)/15
print(math.sqrt(x))