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

# I add this line for testing notification from github to slack-2!

class BackTest(object):
    def __init__(self, strategy, initial_cash, enable_cost=True, enable_db=True):
        self.initial_cash = initial_cash
        self.enable_cost = enable_cost
        self.enable_db = enable_db

        if self.enable_db:
            self.__connect_db()
        else:
            self.benchmark_df = pd.read_pickle("../ML_TA/data/benchmark.pickle").set_index("timestamp")
            self.price_df = pd.read_pickle("../ML_TA/data/price.pickle")
            self.exRight_exDividens_df = pd.read_pickle("../ML_TA/data/exRight_exDividens.pickle").set_index("timestamp")

        self.stgy_df = self.__get_stgy_df(strategy)

    def get_backtest_df(self):
        self.final_df = self.__execute_backtest()
        return self.final_df

    def __get_stgy_df(self, stgy):
        mod_stgy = []
        # 將strategy存進dataframe
        for t in stgy:
            for sl in t["stockList"]:
                mod_stgy.append({"timestamp": t["timestamp"], "stockId": sl["stockId"], "position": sl["position"]})

        mod_stgy_df = pd.DataFrame(mod_stgy).set_index('timestamp')
        stocklist = mod_stgy_df["stockId"].drop_duplicates().values

        # 將部位相對變動量存進df
        temp_dfs = []
        agg_dict = {}
        for s in stocklist:
            temp_df = mod_stgy_df[mod_stgy_df["stockId"] == s]["position"]
            temp_df.name = "position_{}".format(s)
            temp_dfs.append(temp_df)
            agg_dict.update({temp_df.name: sum})
        stgy_df = pd.concat(temp_dfs, axis=1)

        stgy_df.index = self.__check_traded_days(stgy_df.index.to_list())
        # 將平移後的多筆交易和為一筆
        stgy_df = stgy_df.groupby(stgy_df.index).agg(agg_dict)
        return stgy_df

    def __execute_backtest(self):
        multiple_dfs = []
        #針對每隻標的計算獲利、虧損與交易成本
        for col in self.stgy_df.columns:
            startTime = self.stgy_df[col].index[0]
            endTime = self.stgy_df[col].index[-1]
            stgy_on_stock_df = self.stgy_df[col]
            stockId = col.split("position_")[1]
            #計算個股的虧損
            single_df = self.__get_single_asset_df(stockId, startTime, endTime, stgy_on_stock_df)
            multiple_dfs.append(single_df)
        #計算集合獲利與虧損
        final_df = self.__get_multiple_asset_df(multiple_dfs, cash=self.initial_cash)
        benchmark_df = self.__get_db_benchmark_df(startTime=final_df.index[0], endTime=final_df.index[-1])
        final_df = pd.concat([final_df, benchmark_df], axis=1)
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
        finalStrategyDaysArray = np.concatenate((OKstrategyDaysArray, np.array(okedList)), axis=None)

        def npdt64Todatetime(dt64array):
            ts = (dt64array - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            return [datetime.datetime.utcfromtimestamp(t) for t in ts]
        finalStrategyDaysList = sorted(npdt64Todatetime(finalStrategyDaysArray))

        return finalStrategyDaysList

    def __get_single_asset_df(self, stockId, startTime, endTime, stgy_on_stock_df):
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
        #獲取該資料的收盤價
        close_df  = self.__get_db_close_series(stockId, startTime, endTime)
        #收盤價和策略合併
        temp_df = pd.concat([close_df, stgy_on_stock_df], axis=1).fillna(method="ffill").fillna(0)
        #計算場上的投資盈虧
        temp_df["portfolio_{}".format(stockId)] = (temp_df["close"] *
                                                   temp_df["position_{}".format(stockId)]
                                                  ).fillna(method="ffill")

        temp2_series = stgy_on_stock_df.dropna() #找出策略的執行日期
        exe_diff_series = temp2_series.diff() #找出策略的執行日期與當日該變動的數量
        exe_diff_series.iloc[0] = temp2_series.iloc[0] #策略執行第一天，股票從零到有(或做空)
        if self.enable_cost:
            temp_df["abs_cost_{}".format(stockId)] = (exe_diff_series * close_df).apply(self.__parse_FeeAndTax).replace(0, np.nan) #計算交易費用與交易稅
        else:
            temp_df["abs_cost_{}".format(stockId)] = (exe_diff_series * close_df).apply(self.__parse_FeeAndTax) #計算交易費用與交易稅

        temp_df["profit_{}".format(stockId)] = -1 * exe_diff_series * close_df #淨獲利，若是買入股票則是負值；賣出則為正

        #考慮除權息
        try:
            ex_df = self.__get_db_ex_divided(stockId, startTime, endTime)
            ex_df = ex_df.set_index("timestamp")["權值加息值"].apply(lambda x: float(x.replace(",", "")))
            temp_df = pd.concat([temp_df, ex_df.drop_duplicates()], axis=1)
            temp_df["profit_{}".format(stockId)] = (temp_df["profit_{}".format(stockId)].fillna(0) +
                                                        temp_df["權值加息值"].fillna(0) * temp_df["position_{}".format(stockId)]
                                                   ).replace(0, np.nan)
            temp_df = temp_df.drop(columns=["close", "權值加息值"])
        except KeyError:
            temp_df = temp_df.drop(columns=["close"])

        return temp_df

    def __get_multiple_asset_df(self, df_list, cash):
        """
        fn: 將多個從__get_single_asset_df傳回的df整合
        input: df_list: list結構，像是這樣: [{"df": df, "stockId": stockId}, ...]
               cash: 初始資金，float
        output: pd.DataFrame with columns
            ["total_capital", "net_profit", "net_asset_portfolio", "net_cost", "stockId1", "stockId2", ...]
            and datetime as index
        """

        temp_df = pd.concat(df_list, axis=1)

        abs_cost_name_list = [s for s in temp_df.columns if "abs_cost" in s]
        net_profit_name_list = [s for s in temp_df.columns if "profit" in s]
        position_name_list = [s for s in temp_df.columns if "position" in s]
        portfolio_name_list = [s for s in temp_df.columns if "portfolio" in s]

        temp_df[portfolio_name_list] = temp_df[portfolio_name_list].fillna(method="ffill")
        temp_df[position_name_list] = temp_df[position_name_list].fillna(method="ffill")
        if self.enable_cost:
            temp_df["net_cost"] = temp_df[abs_cost_name_list].sum(axis=1).replace(0, np.nan)
        else:
            temp_df["net_cost"] = temp_df[abs_cost_name_list].sum(axis=1)
        temp_df["net_profit"] = temp_df[net_profit_name_list].sum(axis=1)
        temp_df["net_asset_portfolio"] = temp_df[portfolio_name_list].sum(axis=1)
        # print(temp_df)

        #計算現金變動
        cash_list = []
        for t in temp_df["net_cost"].dropna().index:      #只在有交易的時候，現金才會變動
            sub_series = temp_df.loc[t]
            cash = cash + sub_series["net_profit"] - sub_series["net_cost"] #更新現金數量 = 原有現金 + 淨獲利 - 摩擦成本
            cash_list.append({"timestamp": t, "cash":cash}) #紀錄現金

        cash_df = pd.DataFrame(cash_list).set_index('timestamp') #儲存成series
        temp_df = pd.concat([temp_df, cash_df], axis=1) #將現金series合併進temp_df

        temp_df["cash"] = temp_df["cash"].fillna(method="ffill") #把現金中沒有值的地方，用上一個非nan的值填起來
        temp_df["total_capital"] = temp_df["cash"] + temp_df["net_asset_portfolio"] #計算總資產

        temp_df = temp_df[["total_capital", "cash", "net_cost", "net_profit", "net_asset_portfolio"] + position_name_list + portfolio_name_list]

        return temp_df

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

    def __get_db_benchmark_df(self, startTime, endTime):
        if self.enable_db:
            benchmark_df = pd.DataFrame(list(self.db["dailyBenchmarks"].find(
                    {"timestamp": {"$gt": startTime, "$lt": endTime},
                     "stockId":"發行量加權股價指數"}, {"timestamp": 1, "收盤指數": 1}))).drop(columns="_id").drop_duplicates("timestamp").set_index("timestamp")
        else:
            benchmark_df = self.benchmark_df

        benchmark_df["收盤指數"] = benchmark_df["收盤指數"].apply(lambda x:
        	                                             float(x.replace("--", "0").replace('---',"0").replace(",", "")))
        # Renormalize to the initial_cash
        benchmark_df['收盤指數'] = self.initial_cash/benchmark_df['收盤指數'].iloc[0]*benchmark_df['收盤指數']

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
        df = df["close"]
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
                        ))).drop(columns="_id")
        else:
            df = (self.exRight_exDividens_df[self.exRight_exDividens_df["stockId"]==stockId]
                    ).loc[startTime: endTime+ datetime.timedelta(days=10)]
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
                cost = x * fee
            else:
                cost = 0
            return np.abs(cost)
        else: return 0

class FixedInvestmentStrategy(object):
    def __init__(self, stockId, startTime, endTime, initial_cash, annual_interest_rate, freq, position):
        self.stockId = stockId
        self.startTime = startTime
        self.endTime = endTime
        self.initial_cash = initial_cash
        self.annual_interest_rate = annual_interest_rate
        self.freq = freq
        self.position = position

    def df_to_strategy(self, df):
        output_stgy = []
        for t in df.index:
            stockList = []
            for c in df.columns:
                position = df[c].loc[t]
                stockId = c
                stockList.append({"stockId": c, "position": position})
            output_stgy.append({"timestamp": t.to_pydatetime(), "stockList": stockList})
        return output_stgy

    def get_cash_interest_df(self):
        df = pd.DataFrame(index=pd.date_range(start=self.startTime, end=self.endTime, freq=self.freq))
        df["risk_free_cash"] = 0
        df["risk_free_cash"].iloc[0] = self.initial_cash
        for i in range(1, df.shape[0]):
            df["risk_free_cash"].iloc[i] =  df["risk_free_cash"].iloc[i-1] * (1 + self.annual_interest_rate/12)
        return df["risk_free_cash"]

    def get_fixedInvestment_df(self):
        df = pd.DataFrame(index=pd.date_range(start=self.startTime, end=self.endTime, freq=self.freq))
        df[self.stockId] = self.position
        df[self.stockId] = df[self.stockId].cumsum()

        output_stgy = self.df_to_strategy(df)

        cash_df = self.get_cash_interest_df()

        backtest = BackTest(output_stgy, initial_cash = self.initial_cash, enable_db=False)
        output_df = backtest.get_backtest_df()
        output_df = pd.concat([output_df, cash_df], axis=1).fillna(method="ffill")
        return output_df


if __name__ == '__main__':
    startTime = "2004/01/01"
    endTime = "2019/07/30"
    stockId = "0050"
    freq = "1M"
    position = 10
    initial_cash = 1.1e5
    annual_interest_rate = 0.01

    fixedStrategy = FixedInvestmentStrategy(stockId, startTime, endTime, initial_cash, annual_interest_rate, freq, position)
    output_df = fixedStrategy.get_fixedInvestment_df()
    output_df[["total_capital" ,"risk_free_cash"]].plot()
    plt.show()
