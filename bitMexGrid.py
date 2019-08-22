import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import pyfolio as pf
import talib
from talib import MA_Type
from itertools import product
from matplotlib import pyplot as plt
# fig, ax = plt.subplots()

class bitMexGrid(object):

    def __init__(self, stockId, startTime, endTime, totalCash= 1e6, SP = 1.5, SL = 0.5):
        self.stockId = stockId;  self.startTime = startTime;            self.endTime = endTime
        self.totalCash = totalCash;            self.SP      = SP; self.SL = SL
        self.rPFluid = 0.00025; self.df        = self.__get_bitMexHL()

    def get_pf_charts(self):
        """get return and position df for pyfolio to analyze"""
        # self.__gridGridBT()
        # self.df[['total_capital', 'close']].pct_change(1).cumsum().plot()

        # df = self.result
        # self.pf_return_df = df["total_capital"].pct_change(1)
        # self.pf_position_df = df[['position', 'cash']]
        # self.benchmark_df = df["close"].pct_change(1)
        # pf.create_full_tear_sheet(returns=self.pf_return_df,
        #                           positions=self.pf_position_df,
        #                           benchmark_rets=self.benchmark_df)#,
#                                  transactions=self.transactions_df)

    def __gridGridBT(self):
        # self.df['BBWith'] =
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
        self.df['rowNum'] = range(len(self.df))
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
              'timestamp': '1' }))).drop(columns="_id").set_index("timestamp").drop_duplicates()
        # df.columns = ['open', 'close', 'low', 'high']
        return df

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

if __name__ == '__main__':
    bM = bitMexGrid('XBTUSD', pd.Timestamp(2019, 8, 18, 0, 0, 00), pd.Timestamp(2019, 8, 21, 0, 0, 00), totalCash= 1e6)
    bM.get_pf_charts()

test = bM.df.copy()

test.columns = pd.MultiIndex.from_product([['common'], test.columns.values])

nextB= 1.0007
next= 1.001
down = 2-next
T=0.00025
posRatio = 0.1
initial_cash = 1e6

u, m, l = talib.BBANDS(bM.df['close'], matype=MA_Type.T3)

test[('common','m')] = m
test[('common','u')] = u
test[('common','l')] = l

test=test.dropna()

test[('common','rowNum')] = range(len(test))

for i in product(['short', 'long', 'result'], ['position', 'portfolio', 'cash', 'profit', 'total_capital']):
    test[i] = 0

test[('long','numL')] = round((np.log(test[( 'common','u')]/test[('common','m')])/np.log(nextB)).cumsum()/test[( 'common','rowNum')], 0).replace(np.inf, 0)
test[( 'short','numS')] = round((np.log(test[('common','l')]/test[( 'common','m')])/np.log(2-nextB)).cumsum()/test[( 'common','rowNum')], 0).replace(np.inf, 0)

test['short', 'cash'] = initial_cash/2
test['long', 'cash'] = initial_cash/2

maxU = int(test['long', 'numL'].max())
maxL = int(test['short', 'numS'].max())
for i in range(maxU):
    test['long', str(i)] =0.0
for i in range(maxL):
    test['short', str(i)] = 0.0
# first happening rows of the long and short DF could be different.
# from here, we consider them separately.
firstLE = 0; nextLE = 0
firstSE = 0; nextSE = 0
while firstLE < len(test):
    cashRefLE = test[('long','cash')][firstLE-1]
    test.loc[firstLE:, ('long', 'cash')] = cashRefLE
    firstLE = test[('common','rowNum')][test[('long', 'numL')] != 0][firstLE:][0]
    closeRefLE = test[('common','close')][firstLE-1]

    print(closeRefLE, cashRefLE, firstLE)

    test_bL = test[firstLE:].copy()
    test_bL['long', 'Event_L'] = (np.log(test_bL[('common','high')].dropna()/closeRefLE)/np.log(next)).astype(int)
    test_bL['long', 'Event_L'] *= (test_bL['long', 'Event_L'] >= 0)*1
    test_bL['long', 'nextEvent'] = (test_bL['long', 'Event_L'] >= test_bL['long', 'numL'][0])
    test_bL['long', 'cash'] = cashRefLE

    try:
        nextLE = test_bL[test_bL['long', 'nextEvent'] != 0]['common', 'rowNum'][0]
    except:
        nextLE = len(test)

    test_cL = test_bL[:nextLE-firstLE+1].copy()

    for i in range(len(test_cL)):
        test_d = test_cL[i:].copy()
        thisCell = int(test_d['long', 'Event_L'][0])
        thisPrice = closeRefLE*(next**thisCell)
        nextFactor = ~test_d['long', 'nextEvent'][0]*1
        portBuy = test_d['long', 'cash'][0] * ((1-posRatio+posRatio*T+posRatio*next+posRatio*next*T)**thisCell)*posRatio * nextFactor
        posBuy = portBuy / (thisPrice)
        try:
            test_d['long', str(thisCell)]+=posBuy
        except:
            thisCell = maxU

        thisProfit = test_d['long','cash'][0]*(((1-posRatio+posRatio*T+posRatio*next+posRatio*next*T)**thisCell)*(1-posRatio+posRatio*T)-1) * nextFactor
        test_d['long', 'position'] += posBuy
        test_d['long', 'profit'] = thisProfit

        # if triggered, buy/sell previous one-cell position.
        for ii in range(thisCell):
            posSell = test_d['long', str(ii)][0]
            portfSell = posSell*thisPrice
            test_d['long', str(ii)] = 0
            test_d['long', 'position'] -= posSell
            thisProfit += portfSell

        test_d['long', 'portfolio'] = test_d['long', 'position'] * test_d['common', 'close']
        test_d['long', 'cash'] += thisProfit
        test_cL[i:]=test_d.copy()

    test.loc[firstLE:nextLE+1, 'long'] = test_cL['long'].drop(columns=['Event_L', 'nextEvent']).values.copy()

    firstLE = nextLE + 1

test['long', 'total_capital'] = test['long', 'cash']+test['long', 'portfolio']
test['long', 'total_capital'].pct_change(1).cumsum().plot()

test['common','close'].pct_change(1).cumsum().plot()# short position
# while count < 1:
while firstSE < len(test):
    # count +=1
    cashRefSE = test[('short','cash')][firstSE-1]
    test.loc[firstSE:, ('short', 'cash')] = cashRefSE
    firstSE = test[('common','rowNum')][test[('short', 'numS')] != 0][firstSE:][0]
    closeRefSE = test[('common','close')][firstSE-1]

    print(closeRefSE, cashRefSE, firstSE)

    test_bS = test[firstSE:].copy()
    test_bS['short', 'Event_S'] = (np.log(test_bS[('common','low')].dropna()/closeRefSE)/np.log(down)).astype(int)
    test_bS['short', 'Event_S'] *= (test_bS['short', 'Event_S'] >= 0)*1
    test_bS['short', 'nextEvent'] =(test_bS['short', 'Event_S'] >= test_bS['short', 'numS'][0])
    test_bS['short', 'cash'] = cashRefSE

    try:
        nextSE = test_bS[test_bS['short', 'nextEvent'] != 0]['common', 'rowNum'][0]
    except:
        nextSE = len(test)

    test_cS = test_bS[:nextSE-firstSE+1].copy()

    for i in range(len(test_cS)):
        test_d = test_cS[i:].copy()
        thisCell = abs(int(test_d['short', 'Event_S'][0]))
        thisPrice = (closeRefSE*down**thisCell)
        nextFactor = ~test_d['short', 'nextEvent'][0]*1
        portSell = test_d['short', 'cash'] * ((1+posRatio+posRatio*T-posRatio*down+posRatio*down*T)**thisCell)*posRatio * nextFactor
        posSell = portSell / (closeRef*down**thisCell)
        try:
            test_d['short', str(thisCell)]+=posSell
        except:
            thisCell = maxL

        thisProfit = test_d['short','cash'][0]*(((1+posRatio+posRatio*T-posRatio*down+posRatio*down*T)**thisCell)*(1+posRatio+posRatio*T)-1) * nextFactor
        test_d['short', 'position'] -= posSell
        test_d['short', 'profit'] = thisProfit

        # if triggered, buy/sell previous one-cell position.
        for ii in range(thisCell):
            posBuy = test_d['short', str(ii)][0]
            portfBuy = posBuy*thisPrice
            test_d['short', str(ii)] = 0
            test_d['short', 'position'] += posBuy
            thisProfit -= portfBuy

        test_d['short', 'portfolio'] = test_d['short', 'position'] * test_d['common', 'close']
        test_d['short', 'cash'] += thisProfit
        test_cS[i:]=test_d.copy()

    test.loc[firstSE:nextSE+1, 'short'] = test_cS['short'].drop(columns=['Event_S', 'nextEvent']).values.copy()

    firstSE = nextSE +1

test['short', 'total_capital'] = test['short', 'cash']+test['short', 'portfolio']
# test['common','close'].plot()
test['short', 'total_capital'].pct_change(1).cumsum().plot()
