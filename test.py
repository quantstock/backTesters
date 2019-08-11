for i in range(1):
    from backTesters.backtest import BackTest
    import datetime
    from pymongo import MongoClient
    import pandas as pd
simStgy =  [{'stockList': [{'position': 1000, 'stockId': '2330'},
                          {'position': -1000, 'stockId': '2317'}],
            'timestamp': datetime.datetime(2010, 1, 31, 0, 0)},
           {'stockList': [{'position': 7000, 'stockId': '2330'},
                          {'position': -7000, 'stockId': '2317'}],
            'timestamp': datetime.datetime(2018, 2, 1, 0, 0)}]
# mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
# dbName = "stock_data" # database name
# db = MongoClient(mongo_uri)[dbName]
# benchmark_df = pd.DataFrame(list(db["dailyBenchmarks"].find(
#         {"timestamp": {"$gt": datetime.datetime(2007, 1, 31, 0, 0), "$lt": datetime.datetime(2008, 2, 1, 0, 0)},
#          "stockId":"發行量加權股價指數"}, {"timestamp": 1, "收盤指數": 1}))).drop_duplicates("timestamp").set_index("timestamp")
a = BackTest(simStgy, 5e6)
df = a.get_backtest_df()
a.get_pf_charts(df)
for t in simStgy:
    print(t)
