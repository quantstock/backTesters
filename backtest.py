"""
backtest modulus
author: wenping lo
last updated: 2019/6/11
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys
import pyfolio as pf
from functools import reduce

# I add this line for testing notification from github to slack!

class BackTest(object):

    def __init__(self, strategy, initial_cash, enable_cost=True, enable_db=True, bM='發行量加權股價指數'):
        self.initial_cash = initial_cash
        self.enable_cost = enable_cost
        self.enable_db = enable_db
        self.BM_Name = bM

        if self.enable_db:
            self.__connect_db()
        else:
            self.benchmark_df = pd.read_pickle("../ML_TA/data/benchmark.pickle").set_index("timestamp")
            self.price_df = pd.read_pickle("../ML_TA/data/price.pickle")
            self.exRight_exDividens_df = pd.read_pickle("../ML_TA/data/exRight_exDividens.pickle").set_index("timestamp")

        self.stgy_df = self.__get_stgy_df(strategy)
        self.final_df = None

    def get_backtest_df(self):
        self.final_df = self.__execute_backtest()
        return self.final_df

    def get_pf_charts(self, df):
        """get return and position df for pyfolio to analyze"""
        self.pf_return_df = df['Results']["total_capital"].pct_change(1)
        pf_position_dfList = [s for s in a.columns if 'portfolio'==s[1]]
        self.pf_position_df = df[pf_position_dfList + [('Results', "cash")]]
        self.pf_position_df.columns = [s[0] for s in pf_position_dfList] + ['cash']
        self.benchmark_df = df["benchmark"][self.BM_Name].pct_change(1)
        pf.create_full_tear_sheet(returns=self.pf_return_df,
                                  positions=self.pf_position_df,
                                  benchmark_rets=self.benchmark_df)#,
#                                  transactions=self.transactions_df)

    def __get_transactions_df(self):
        """將交易成本的細項合併成df, 以供後續pyfolio使用"""
        return pd.concat(self.multiple_trans_dfs, axis=0)

    def __get_stgy_df(self, stgy_df):

        agg_dict = {}

        for si in stgy_df.columns.levels[0]:
            agg_dict.update({(si, 'position'): sum})

        stgy_df.index = self.__check_traded_days(stgy_df.index.to_list())
        # 將平移後的多筆交易和為一筆
        stgy_df = stgy_df.groupby(stgy_df.index).agg(agg_dict)
        self.stgy_df = stgy_df
        return stgy_df

    def __execute_backtest(self):
        singDf_List = self.__get_single_asset_df()
        final_df = self.__get_multiple_asset_df(singDf_List)
        # #
        benchmark_df = self.__get_db_benchmark_df(startTime=self.stgy_df.index[0], endTime=self.stgy_df.index[-1])
        benchmark_df.columns = pd.MultiIndex.from_tuples([('benchmark', self.BM_Name)])
        self.final_df= final_df
        final_df = pd.concat([final_df, benchmark_df], axis=1)
        # final_df = final_df.loc[final_df.xs('position', level=1, axis=1).isnull().prod(axis=1) == 0]
        return final_df

    def __check_traded_days(self, timestampList):
        """
        input: timestamp list, type: python list
        output: timestamp list, type: python list
        function: 檢查策略中的時間戳記是否是交易日，若否，則往後推延直到是交易日
        """
        strategyDaysArray = pd.to_datetime(timestampList).values
        tradindDays_df = self.__get_db_tradingDays_series()
        tradindDays = tradindDays_df.T.values
        existedDaysArray = pd.to_datetime(tradindDays).values[0]
        mask = np.in1d(strategyDaysArray, existedDaysArray)
        notOKstrategyDaysArray = pd.to_datetime(strategyDaysArray[~mask]).values
        OKstrategyDaysArray = pd.to_datetime(strategyDaysArray[mask]).values
        okedList = []
        while notOKstrategyDaysArray.tolist(): #有非交易日存在
            modTradingDayList = []
            for notTradingDay in notOKstrategyDaysArray:
                modTradingDayList.append(notTradingDay + np.timedelta64(1,'D'))
            modTradingDayList = np.array(modTradingDayList)
            mask = np.in1d(modTradingDayList, existedDaysArray)
            notOKstrategyDaysArray = pd.to_datetime(modTradingDayList[~mask]).values
            OKedStrategyDaysArray = pd.to_datetime(modTradingDayList[mask]).values
            for d in OKedStrategyDaysArray:
                okedList.append(d)
        try:
            finalStrategyDaysArray = np.concatenate((OKstrategyDaysArray, np.array(okedList)), axis=None)
        except:
            finalStrategyDaysArray = OKstrategyDaysArray

        def npdt64Todatetime(dt64array):
            ts = (dt64array - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            return [datetime.datetime.utcfromtimestamp(t) for t in ts]
        finalStrategyDaysList = sorted(npdt64Todatetime(finalStrategyDaysArray))

        return finalStrategyDaysList

    # def __get_single_asset_df(self, stockId, startTime, endTime, stgy_on_stock_df):
    def __get_single_asset_df(self):
        """
        fn: 取得個別資產在策略的起始與終止期間的獲利(或虧損)
        input:
            stockId: str
            starTime: datetime
            endTime: datetime
            stgy_on_stock_df: pd.series with timestamp as index, positions as values
        output:
            pd.DataFrame with columns: ["position_stockId", "net_profit_stockId", "portfolio_stockId", "abs_cost_stockId"]
                and datetime as index
        """
        df_List = []
        for col in self.stgy_df.columns:
            startTime = self.stgy_df[col].index[0]
            endTime = self.stgy_df[col].index[-1]
            stgy_on_stock_df = self.stgy_df[col]
            stockId = col[0]
            df = pd.concat([self.__get_db_close_series(stockId, startTime, endTime).drop_duplicates(keep="first"), self.stgy_df[stockId]], axis=1)  #獲取該資料的收盤價

            df['portfolio']=df['position'] * df['close'] #計算場上的投資盈虧
            temp2_series = df['position'].dropna() #找出策略的執行日期
            exe_diff_series = temp2_series.diff() #找出策略的執行日期與當日該變動的數量
            exe_diff_series.iloc[0] = temp2_series.iloc[0] #策略執行第一天，股票從零到有(或做空)
            if self.enable_cost:
                df["abs_cost"] = (exe_diff_series * df["close"]).apply(self.__parse_FeeAndTax).replace(0, np.nan) #計算交易費用與交易稅
            else:
                df["abs_cost"] = (exe_diff_series * df["close"]).apply(self.__parse_FeeAndTax)

            df["profit"] = -1 * exe_diff_series * df["close"] #淨獲利，若是買入股票則是負值；賣出則為正

            #考慮除權息
            ex_df = self.__get_db_ex_divided(stockId, startTime, endTime)
            df=pd.concat([df, ex_df], axis = 1)
            df["profit"] = (df["profit"].fillna(0) + df["權值加息值"].fillna(0)* df["position"]).replace(0, np.nan)
            df = df.drop(columns=["權值加息值"])
            df.columns = pd.MultiIndex.from_product([[stockId], df.columns])
            self.df = df
            df_List.append(df.dropna())

        self.df_List = df_List
        return df_List


    # def __get_multiple_asset_df(self, df_list, cash):
    def __get_multiple_asset_df(self, singDf_List):
        """
        fn: 將多個從__get_single_asset_df傳回的df整合
        input: df_list: list結構，像是這樣: [{"df": df, "stockId": stockId}, ...]
               cash: 初始資金，float
        output: pd.DataFrame with columns
            ["total_capital", "net_profit", "net_asset_portfolio", "net_cost", "stockId1", "stockId2", ...]
            and datetime as index
        """
        resultCol = ['net_cost', 'net_asset_portfolio', 'net_profit', 'cash', 'total_capital']
        singDf_List = reduce(self.__red_PD_Concat, singDf_List).fillna(0) # concat small dfs together
        sumDf = singDf_List.sum(axis=1, level=1)[['abs_cost', 'portfolio', 'profit']] # sum of things to be sum
        g = -sumDf['abs_cost'] + sumDf['profit']
        sumDf['cash'] = g.cumsum() + self.initial_cash # roll over the profit and add to the cash
        sumDf["total_capital"] = sumDf["cash"] + sumDf["portfolio"]
        sumDf.columns = pd.MultiIndex.from_product([['Results'], resultCol])

        return pd.concat([sumDf, singDf_List], axis=1)

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

    def __get_db_benchmark_df(self, startTime, endTime):
        if self.enable_db:
            try:
                benchmark_df = pd.DataFrame(list(self.db["dailyBenchmarks"].find(
                        {"timestamp": {"$gt": startTime, "$lt": endTime},
                         "stockId":self.BM_Name}, {"timestamp": 1, "收盤指數": 1}))).drop(columns="_id").drop_duplicates("timestamp").set_index("timestamp")
                benchmark_df["收盤指數"] = benchmark_df["收盤指數"].apply(lambda x:
                	                                             float(x.replace("--", "0").replace('---',"0").replace(",", "")))
            except:
                benchmark_df = pd.DataFrame(list(self.db["dailyPrice"].find(
                        {"timestamp": {"$gt": startTime, "$lt": endTime},
                         "stockId":self.BM_Name}, {"timestamp": 1, "收盤價": 1}))).drop(columns="_id").drop_duplicates("timestamp").set_index("timestamp")
                benchmark_df["收盤價"] = benchmark_df["收盤價"].apply(lambda x:
                	                                             float(x.replace("--", "0").replace('---',"0").replace(",", "")))
        else:
            benchmark_df = self.benchmark_df

        # Renormalize to the initial_cash
# <<<<<<< bt_stat
        # benchmark_df['收盤指數'] = self.initial_cash/benchmark_df['收盤指數'].iloc[0]*benchmark_df['收盤指數']
# =======
#         benchmark_df['收盤指數'] = self.initial_cash/benchmark_df['收盤指數'].iloc[0]*benchmark_df['收盤指數']
# >>>>>>> master

        benchmark_df = benchmark_df.rename(columns = {"收盤指數" : "benchmark"}).loc[startTime: endTime]
        return benchmark_df

    def __get_db_close_series(self, stockId, startTime, endTime):
        if self.enable_db:
            df = pd.DataFrame(list(self.db["dailyPrice"].find(
                     {"timestamp": {
                         "$gte": startTime, "$lte": endTime + datetime.timedelta(days=10)},
                         "stockId":stockId},  # selection criterion
                     {"timestamp": "-1" ,
                      "收盤價": "1",
                      "最後揭示賣價": "1",
                      "最後揭示買價": "1"}))).drop(columns="_id").drop_duplicates("timestamp", keep="first").set_index("timestamp")

        else:
            df = self.price_df[self.price_df["stockId"]==stockId]
            df = df.drop_duplicates("timestamp", keep="first").set_index("timestamp")
            df = df[["收盤價", "最後揭示賣價", "最後揭示買價"]].loc[startTime: endTime]

        df["close"] = df.apply(self.__parse_close, axis=1)
        df = df[["close"]]
        return df

    def __get_db_tradingDays_series(self):
        """d """
        if self.enable_db:
            # print(pd.DataFrame(list(self.db["tradingDays"].find({}))).drop(columns ="_id"))
            # sys.exit()
            return pd.DataFrame(list(self.db["tradingDays"].find({}))).drop(columns ="_id")
        else:
            # print(self.price_df[self.price_df["stockId"]=="2330"].drop_duplicates("timestamp", keep="first")[["timestamp"]])
            # sys.exit()
            return self.price_df[self.price_df["stockId"]=="2330"].drop_duplicates("timestamp", keep="first")[["timestamp"]]

    def __get_db_ex_divided(self, stockId, startTime, endTime):
        if self.enable_db:
            df = pd.DataFrame(list(self.db["exRight_exDividens"].find(
                            {"timestamp": {
                            "$gte": startTime, "$lte": endTime + datetime.timedelta(days=10)},
                            "stockId":stockId}
                        ))).drop(columns="_id")[['timestamp', '權值加息值']].drop_duplicates("timestamp", keep="first")
        else:
            df = (self.exRight_exDividens_df[self.exRight_exDividens_df["stockId"]==stockId]).loc[startTime: endTime+ datetime.timedelta(days=10)]
        df = df.set_index('timestamp')
        df['權值加息值'] = df['權值加息值'].apply(lambda x: float(x.replace(",", "")))
        return df

    def __parse_close(self, x):
        try: return pd.to_numeric(x["收盤價"].replace(",", ""))
        except ValueError:
            try: return (pd.to_numeric(x["最後揭示賣價"].replace(",", "")) + pd.to_numeric(x["最後揭示賣價"].replace(",", "")))/2
            except ValueError:
                try: return pd.to_numeric(x["最後揭示賣價"].replace(",", ""))
                except ValueError:
                    try: return pd.to_numeric(x["最後揭示買價"].replace(",", ""))
                    except ValueError:
                        return np.nan

    def __parse_FeeAndTax(self, x):
        if self.enable_cost:
            fee = 0.1425*0.01
            tax = 0.3*0.01
            if x < 0:  # short, both fee and tax
                cost = np.minimum(-20, x * fee) + x * tax
            elif x > 0: # long, fee only
                # cost = x * fee
                cost = np.maximum(20, x * fee)
            else:
                cost = 0
            return np.abs(cost)
        else:
            return 0

    def __red_PD_Concat(self, a, b):
        return pd.concat([a, b], axis=1)


class FixedInvestmentStrategy(object):
    def __init__(self, stockId_list, startTime, endTime, initial_cash, annual_interest_rate, freq, position_list, cash_delta):
        self.stockId_list = stockId_list
        self.startTime = startTime
        self.endTime = endTime
        self.initial_cash = initial_cash
        self.annual_interest_rate = annual_interest_rate
        self.freq = freq
        self.position_list = position_list
        self.cash_delta = cash_delta

    def df_to_strategy(self, df):
        output_stgy = []
        for t in df.index:
            stockList = []
            for stockId in df.columns:
                print(stockId)
                position = df[stockId].loc[t]
                stockList.append({"stockId": stockId, "position": position})
            output_stgy.append({"timestamp": t.to_pydatetime(), "stockList": stockList})
        return output_stgy

    def get_cash_interest_df(self):
        df = pd.DataFrame(index=pd.date_range(start=self.startTime, end=self.endTime, freq=self.freq))
        df["risk_free_cash"] = 0
        df["risk_free_cash"].iloc[0] = self.initial_cash

        for i in range(1, df.shape[0]):
            df["risk_free_cash"].iloc[i] =  df["risk_free_cash"].iloc[i-1] * (1 + self.annual_interest_rate/2)+ self.cash_delta
        return df["risk_free_cash"]

    def get_fixedInvestment_df(self):
        df = pd.DataFrame(index=pd.date_range(start=self.startTime, end=self.endTime, freq=self.freq))
        for stockId, position in zip(self.stockId_list, self.position_list):
            df[stockId] = position
            df[stockId] = df[stockId].cumsum()

        output_stgy = self.df_to_strategy(df)

        cash_df = self.get_cash_interest_df()

        backtest = BackTest(output_stgy, initial_cash = self.initial_cash, enable_db=True)
        # output_df = backtest.get_backtest_df()
        # output_df = pd.concat([output_df, cash_df], axis=1).fillna(method="ffill")
        return backtest

if __name__ == '__main__':
    startTime = "2010/01/01"
    endTime = "2019/08/04"
    stockId_list = ["2330", "1101"]
    freq = "6M"
    position_list = [5000, -50000]
    initial_cash = 1e7
    # cash_delta = 60000
    annual_interest_rate = 0.01

    fixedStrategy = FixedInvestmentStrategy(stockId_list, startTime, endTime, initial_cash, annual_interest_rate, freq, position_list, cash_delta)
    bt = fixedStrategy.get_fixedInvestment_df()
    df = bt.get_backtest_df()
    bt.get_pf_charts()
