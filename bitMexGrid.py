import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import pyfolio as pf

class bitMexGrid(object):

    def __init__(self, stockId, startTime, endTime, totalCash= 1e6, SP = 1.5, SL = 0.5):
        self.stockId = stockId;  self.startTime = startTime;            self.endTime = endTime
        self.totalCash = totalCash;            self.SP      = SP; self.SL = SL
        self.rPFluid = 0.00025; self.df        = self.__get_bitMexHL()

    def get_pf_charts(self):
        """get return and position df for pyfolio to analyze"""
        self.__gridGridBT()
        self.df[['total_capital', 'close']].pct_change(1).cumsum().plot()

        # df = self.result
        # self.pf_return_df = df["total_capital"].pct_change(1)
        # self.pf_position_df = df[['position', 'cash']]
        # self.benchmark_df = df["close"].pct_change(1)
        # pf.create_full_tear_sheet(returns=self.pf_return_df,
        #                           positions=self.pf_position_df,
        #                           benchmark_rets=self.benchmark_df)#,
#                                  transactions=self.transactions_df)

    def __gridGridBT(self):
        self.df['SellEvent'] = (self.df['high'] == self.df['high'].cummax()) & (np.log(self.df['high']/self.df['open'])/np.log(1.0025) >= 1 )
        self.df['BuyEvent'] = (self.df['low'] == self.df['low'].cummin()) & (np.log(self.df['low']/self.df['open'])/np.log(2-1.0025) >= 1 )
        self.df['high'] *= self.df['SellEvent'].fillna(0)
        self.df['low'] *= self.df['BuyEvent'].fillna(0)
        self.df['BuyEvent'] *= -1
        self.df['SellEvent'] *= 1
        self.df['Event'] = self.df['BuyEvent'] + self.df['SellEvent']
        self.df['EventPrice'] = self.df['high'] + self.df['low'] + self.df['close'] * (self.df['Event'] == 0)*1
        self.df = self.df.drop(columns =['BuyEvent', 'SellEvent'])

        T = 1/(2-self.rPFluid)
        firstPos = T*self.totalCash/self.df['open'][0]

        self.df['position'] = firstPos
        self.df['portfolio'] = firstPos*self.df['EventPrice']
        self.df['cash'] = self.df['portfolio'][0] * (1+self.rPFluid)
        self.df['rowNum'] = range(len(a))
        rowNum = self.df['rowNum'][self.df['Event'] != 0]
        self.df = self.df.drop(columns ='rowNum')
        self.df['profit'] = 0

        for i in rowNum.values:
            b=self.df[i:].copy()
            b['profit'] = b['Event']*abs(b['portfolio'] - b['cash'])*(1/(2 + b['Event']*self.rPFluid))
            profit = b['profit'][0]
            b['cash'] += profit + abs(profit)*self.rPFluid
            b['portfolio'] -= profit
            b['position'] += profit/b['EventPrice']
            self.df[i:] = b.copy()
        self.df['position'][0] = firstPos; self.df['position']=self.df['position'].replace(np.inf, np.nan).fillna(method='ffill')
        self.df['total_capital'] = self.df['cash'] + self.df['portfolio']

    def __get_bitMexHL(self):
        db = self.__connect_db()
        df = pd.DataFrame(list(self.db['bitmex_1m'].find({"timestamp": {
            "$gte": self.startTime, "$lte": self.endTime},
            "stockId":self.stockId},
            { "open": "1",
              "close": "1",
              "low": "1",
              "high": "1",
              "": "1",
              'timestamp': '1' }))).drop(columns="_id").set_index("timestamp").drop_duplicates()
        return df

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

if __name__ == '__main__':
    bM = bitMexGrid('XBTUSD', pd.Timestamp(2017, 1, 1, 0, 0, 00), pd.Timestamp(2017, 8, 19, 0, 0, 00), totalCash= 1e6)
    bM.get_pf_charts()
