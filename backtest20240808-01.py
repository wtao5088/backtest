import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from itertools import product
import pandas as pd
import dask.dataframe as dd
import backtrader as bt
import json
import time
from datetime import timedelta, datetime
from collections import OrderedDict

# def disable_logging():
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.CRITICAL)
#     while root_logger.handlers:
#         root_logger.handlers.pop()
#     logging.captureWarnings(True)
#     # 禁用所有子记录器
#     for logger_name in logging.root.manager.loggerDict:
#         logger = logging.getLogger(logger_name)
#         logger.setLevel(logging.CRITICAL)
#         logger.handlers = []
#         logger.propagate = False
#
# # 调用函数来关闭日志系统
# disable_logging()

class AutoDict(dict):
    def _close(self):
        self._closed = True
        print(f"Closing AutoDict: {id(self)}")
        for key, val in self.items():
            if isinstance(val, (AutoDict, AutoOrderedDict)):
                print(f"Closing nested dict for key: {key}")
                val._close()

class AutoOrderedDict(OrderedDict):
    def _close(self):
        self._closed = True
        print(f"Closing AutoOrderedDict: {id(self)}")
        for key, val in self.items():
            if isinstance(val, (AutoDict, AutoOrderedDict)):
                print(f"Closing nested ordered dict for key: {key}")
                val._close()

class QQEMODStrategy(bt.Strategy):
    strategy_params = {
        'rsi1_period': 21,
        'rsi2_period': 34,
        'sf1': 5,
        'sf2': 5,
        'qqe1': 1.618,
        'qqe2': 5.618,
        'bb_length': 50,
        'bb_mult': 1,
        'trade_size': 0.01,
        'atr_multiplier': 5,
        'atr_period': 14,
    }
    def __init__(self, **strategy_params):
        self.strategy_params.update(strategy_params)
        self.rsi1_period = strategy_params['rsi1_period']
        self.rsi2_period = strategy_params['rsi2_period']
        self.sf1 = strategy_params['sf1']
        self.sf2 = strategy_params['sf2']
        self.qqe1 = strategy_params['qqe1']
        self.qqe2 = strategy_params['qqe2']
        self.bb_length = strategy_params['bb_length']
        self.bb_mult = strategy_params['bb_mult']
        self.trade_size = strategy_params['trade_size']
        self.atr_multiplier = strategy_params['atr_multiplier']
        self.atr_period = strategy_params['atr_period']
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.close_prices = []
        self.high_prices = []
        self.low_prices = []
        self.trend1 = 0
        self.trend2 = 0
        self.stop_loss = 0.0
        self.newshortband1 = []
        self.newlongband1 = []
        self.newshortband2 = []
        self.newlongband2 = []
        self.longband1 = None
        self.shortband1 = None
        self.longband2 = None
        self.shortband2 = None
        self.rsi1 = bt.indicators.RSI_SMA(self.datas[0].close, period=self.rsi1_period)
        self.rsi2 = bt.indicators.RSI_SMA(self.datas[0].close, period=self.rsi2_period)
        self.ema_rsi1 = bt.indicators.ExponentialMovingAverage(self.rsi1, period=self.sf1)
        self.ema_rsi2 = bt.indicators.ExponentialMovingAverage(self.rsi2, period=self.sf2)
        self.dar1 = []
        self.dar2 = []
        self.atr1 = []
        self.atr2 = []
        self.atr = []
        self.cross_up1 = []
        self.cross_up2 = []
        self.cross_11 = []
        self.cross_12 = []
        self.order = None
        self.buy_signal = False
        self.sell_signal = False
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.atr_period)
        self.stop_loss1 = self.atr * self.atr_multiplier
        self.fast1 = 0.0
        self.fast2 = 0.0
        self.fast11 = self.fast1 - 50
        self.fast21 = self.fast1 - 50
        self.basic1 = bt.indicators.SmoothedMovingAverage(self.fast11, period=self.bb_length)
        self.basic2 = bt.indicators.SmoothedMovingAverage(self.fast21, period=self.bb_length)
        self.dev1 = self.bb_mult * bt.indicators.MeanDeviation(self.fast11, self.basic1)
        self.dev2 = self.bb_mult * bt.indicators.MeanDeviation(self.fast21, self.basic2)
        self.upper1 = self.basic1 + self.dev1
        self.upper2 = self.basic2 + self.dev2
        self.lower1 = self.basic1 - self.dev1
        self.lower2 = self.basic2 - self.dev2
        self.greenbar1 = 0
        self.greenbar2 = 0
        self.redbar1 = 0
        self.redbar2 = 0

    def next(self):
        current_date = self.datas[0].datetime.datetime(0)
        print(f"current_date: {current_date}, self.dataclose[0]: {self.dataclose[0]}")
        current_close = self.datas[0].close[0]
        self.close_prices.append(current_close)
        self.high_prices.append(self.datas[0].high[0])
        self.low_prices.append(self.datas[0].low[0])
        required_data_length = max(self.rsi1_period, self.rsi2_period, self.sf1, self.sf2) + 1
        if len(self.close_prices) < required_data_length:
            return
        recent_close_prices = self.close_prices[-required_data_length:]
        self.rsi1[0] = bt.indicators.RSI_SMA(self.datas[0].close[0], period=self.rsi1_period)
        self.rsi2[0] = bt.indicators.RSI_SMA(self.datas[0].close[0], period=self.rsi2_period)
        if len(self.rsi1) < required_data_length or len(self.rsi1) < required_data_length:
            return
        self.ema_rsi1[0] = bt.indicators.ExponentialMovingAverage(bt.feeds.DataBase(dataname=self.rsi1), period=self.sf1)
        self.ema_rsi2[0] = bt.indicators.ExponentialMovingAverage(bt.feeds.DataBase(dataname=self.rsi1), period=self.sf2)
        if len(self.ema_rsi1) < required_data_length or len(self.ema_rsi2) < required_data_length:
            return
        atr1 = abs(self.ema_rsi1[-2] - self.ema_rsi1[-1])
        atr2 = abs(self.ema_rsi2[-2] - self.ema_rsi2[-1])
        self.atr1.append(atr1)
        self.atr2.append(atr2)
        if len(self.atr1) < 3 or len(self.atr2) < 3:
            return
        dar1 = self.atr1[0] * self.qqe1
        dar2 = self.atr2[0] * self.qqe2
        self.dar1.append(dar1)
        self.dar2.append(dar2)
        newshortband1 = self.ema_rsi1[0] + self.dar1[0]
        newlongband1 = self.ema_rsi1[0] - self.dar1[0]
        newshortband2 = self.ema_rsi2[0] + self.dar2[0]
        newlongband2 = self.ema_rsi2[0] - self.dar2[0]
        self.newshortband1.append(newshortband1)
        self.newlongband1.append(newlongband1)
        self.newshortband2.append(newshortband2)
        self.newlongband2.append(newlongband2)
        if self.longband1 is None:
           self.longband1 = []
           self.longband1.append(newlongband1)
        else:
            if self.longband1[-1] is not None and self.newlongband1[0] is not None:
                if (self.rsi1[-1] > self.longband1[-1]) and (self.rsi1[-2] > self.longband1[-1]):
                    new_value = max(self.longband1[-1], newlongband1)
                else:
                    new_value = newlongband1
                self.longband1.append(new_value)
        if self.shortband1 is None:
           self.shortband1 = []
           self.shortband1.append(newshortband1)
        else:
            if self.shortband1[-1] is not None and newshortband1 is not None:
               if (self.rsi1[-1] < self.shortband1[-1]) and (self.rsi1[-2] < self.shortband1[-1]):
                   new_value = min(self.shortband1[-1], newshortband1)
               else:
                   new_value = newshortband1
               self.shortband1.append(new_value)
        if self.longband2 is None:
           self.longband2 = []
           self.longband2.append(newlongband2)
        else:
            if self.longband2[-1] is not None and self.newlongband2[0] is not None:
                if (self.rsi2[-1] > self.longband2[-1]) and (self.rsi2[-2] > self.longband2[-1]):
                    new_value = max(self.longband2[-1], newlongband2)
                else:
                    new_value = newlongband2
                self.longband2.append(new_value)
        if self.shortband2 is None:
            self.shortband2 = []
            self.shortband2.append(newshortband2)
        else:
            if self.shortband2[-1] is not None and self.newshortband2[0] is not None:
                if (self.rsi2[-1] < self.shortband2[-1]) and (self.rsi2[-2] < self.shortband2[-1]):
                    new_value = min(self.shortband2[-1], newshortband2)
                else:
                    new_value = newshortband2
                self.shortband2.append(new_value)
        if len(self.shortband1) < 3 or len(self.shortband2) < 3 :
            return
        if self.rsi1[-1] <= self.shortband1[-2] and self.rsi1[0] > self.shortband1[-1]:
            print("RSI1 crossed above shortband1")
            self.trend1 = 1
            if self.fast11 == 0.0:
                self.fast11 = []
                self.fast11.append(self.longband1[0])
            else:
                self.fast11[0] = self.longband1[0]
        elif self.rsi1[-1] >= self.shortband1[-2] and self.rsi1[0] < self.shortband1[-1]:
            self.trend1 = -1
            if self.fast11 == 0.0:
                self.fast11 = []
                self.fast11.append(self.shortband1[0])
            else:
                self.fast11[0] = self.shortband1[0]
        else:
            self.trend1 = getattr(self, 'trend1', 0)
            if self.fast11 == 0.0:
                self.fast11 = []
                self.fast11.append(self.shortband1[0])
            else:
                self.fast11[0] = self.shortband1[0]
        if self.rsi2[-1] <= self.shortband2[-2] and self.rsi2[0] > self.shortband2[-1]:
            print("RSI2 crossed above shortband2")
            self.trend2 = 1
            if self.fast21 == 0.0:
                self.fast21 = []
                self.fast21.append(self.longband2[0])
            else:
                self.fast21[0] = self.longband2[0]
        elif self.rsi2[-1] >= self.shortband2[-2] and self.rsi2[0] < self.shortband2[-1]:
            self.trend2 = -1
            if self.fast21 == 0.0:
                self.fast21 = []
                self.fast21.append(self.shortband2[0])
            else:
                self.fast21[0] = self.shortband2[0]
        else:
            self.trend1 = getattr(self, 'trend1', 0)
            if self.fast21 == 0.0:
                self.fast21 = []
                self.fast21.append(self.shortband2[0])
            else:
                self.fast21[0] = self.shortband2[0]
        if len(self.fast1) < 3 or len(self.fast2) < 3:
            return
        self.fast11.append(abs(self.fast1[0] - 50))
        self.fast21.append(abs(self.fast2[0] - 50))
        fast11 = self.fast11[0]
        fast21 = self.fast21[0]
        if len(self.fast11) < 3 or len(self.fast21) < 3:
            return
        bb_length = self.bb_length
        self.basic1 = bt.indicators.SmoothedMovingAverage(self.fast11, period=self.bb_length)
        self.basic2 = bt.indicators.SmoothedMovingAverage(self.fast21, period=self.bb_length)
        self.basic1 = bt.indicators.SmoothedMovingAverage(self.fast11, period=self.bb_length)
        self.basic2 = bt.indicators.SmoothedMovingAverage(self.fast21, period=self.bb_length)
        basic1_value = bt.indicators.SmoothedMovingAverage(fast11[0], period=bb_length)
        basic2_value = bt.indicators.SmoothedMovingAverage(fast21[0], period=bb_length)
        self.basic1.append(basic1_value)
        self.basic2.append(basic2_value)
        if len(self.basic1) < 2 or len(self.basic2) < 2 :
            return
        bb_mult = self.strategy_params['bb_mult']
        self.dev1 = bb_mult * bt.indicators.MeanDeviation(fast11, basic1_value)
        self.dev2 = bb_mult * bt.indicators.MeanDeviation(fast21, basic2_value)
        dev1 = self.dev1[0]
        dev2 = self.dev2[0]
        self.dev1.append(dev1)
        self.dev2.append(dev2)
        upper1 = basic1_value + dev1
        lower1 = basic1_value - dev1
        self.upper1.append(upper1)
        self.lower1.append(lower1)
        upper2 = basic2_value + dev2
        lower2 = basic2_value - dev2
        self.upper2.append(upper2)
        self.lower2.append(lower2)
        ema_rsi1_value = self.ema_rsi1[0]
        upper1_value = self.upper1[0]
        lower1_value = self.lower1[0]
        ema_rsi2_value = self.ema_rsi2[0]
        upper2_value = self.upper2[0]
        lower2_value = self.lower2[0]
        if (ema_rsi1_value - 50) > upper1_value:
            self.greenbar1 = 1
        else:
            self.greenbar1 = 0
        if ema_rsi2_value - 50 > upper2_value:
            self.greenbar2 = 1
        else:
            self.greenbar2 = 0
        if  ema_rsi1_value - 50 < lower1_value:
            self.redbar1 = 1
        else:
            self.redbar1 = 0
        if ema_rsi2_value - 50 < lower2_value:
            self.redbar1 = 1
        else:
            self.redbar1 = 0
        buy_signal = self.trend1 == 1 and self.trend2 == 1 and self.greenbar1 == 1 and self.greenbar2 == 1
        sell_signal = self.trend1 == -1 and self.trend2 == -1 and self.redbar1 == 1 and self.redbar2 == 1
        if buy_signal:
            if self.position.size < 0:
                self.close()
                self.buy(size=self.trade_size)
                self.stop_loss = self.low_prices[-1] - self.stop_loss1[-1]
                self.trend1 = 0
                self.trend2 = 0
            elif self.position.size >= 0:
                self.buy(size=self.trade_size)
                self.stop_loss = self.low_prices[-1] -self.stop_loss1[-1]
                self.trend1 = 0
                self.trend2 = 0
        elif sell_signal:
            if self.position.size > 0:
                self.close()
                self.sell(size=self.trade_size)
                self.stop_loss = self.high_prices[-1] + self.stop_loss1[-1]
                self.trend1 = 0
                self.trend2 = 0
            elif self.position.size <= 0:
                self.sell(size=self.trade_size)
                self.stop_loss = self.high_prices[-1] + self.stop_loss1[-1]
                self.trend1 = 0
                self.trend2 = 0

def notify_trade(self, trade):
    if not trade.isclosed:
        return

def notify_order(self, order):
    if order.status in [order.Completed]:
        if order.isbuy():
            print(f"Buy order executed, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}")
        elif order.issell():
            print(f"Sell order executed, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}")

def chunked(iterable, size):
    iterable = list(iterable)
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def load_data(file_path):
    data = dd.read_csv(file_path, parse_dates=['datetime'])
    data = data.set_index('datetime')
    data = data.compute()
    return data

def run_backtest_process(**strategy_params):
    rsi1_period = strategy_params['rsi1_period']
    rsi2_period = strategy_params['rsi2_period']
    sf1 = strategy_params['sf1']
    sf2 = strategy_params['sf2']
    qqe1 = strategy_params['qqe1']
    qqe2 = strategy_params['qqe2']
    bb_length = strategy_params['bb_length']
    bb_mult = strategy_params['bb_mult']
    trade_size = strategy_params['trade_size']
    atr_multiplier = strategy_params['atr_multiplier']
    atr_period = strategy_params['atr_period']

    if not all(param in strategy_params for param in ['rsi1_period', 'rsi2_period', 'sf1', 'sf2', 'qqe1', 'qqe2']):
        return

    data_chunk = strategy_params.pop('data_chunk', None)
    if data_chunk is None:
        raise ValueError("data_chunk 参数是必须的")

    required_data_length = max(
        strategy_params['rsi1_period'],
        strategy_params['rsi2_period'],
        strategy_params['sf1'],
        strategy_params['sf2']
    ) + 1

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=data_chunk)
    cerebro.adddata(data)

    cerebro.addstrategy(QQEMODStrategy, **strategy_params)

    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.05)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')

    start_time = time.time()
    strat = cerebro.run()
    end_time = time.time()

    result = strat[0].analyzers.TradeAnalyzer.get_analysis()
    drawdown = strat[0].analyzers.DrawDown.get_analysis().max.DrawDown
    trades = result.get('trades', [])

    if trades:
        net_profits = [trade.pnl.net for trade in trades if 'pnl' in trade]
    else:
        net_profits = []

    total_net_profit = sum(net_profits)
    percent_profitable = (total_net_profit / 10000) * 100 if 10000 else 0
    total_closed_trades = result.total.closed if 'total' in result and 'closed' in result.total else 0
    won_trades = result.won.total if 'won' in result and 'total' in result.won else 0
    percent_won = (won_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0

    result_data = {
        'net_profit': total_net_profit,
        'percent_profitable': percent_profitable,
        'max_drawdown': drawdown,
        'total_close_trade': total_closed_trades,
        'won_trades': won_trades,
        'percent_won': percent_won,
        'strategy_params': strategy_params
    }
    return result_data

def optimize_strategy_params(data, start_date, end_date, param_chunk_size=100, n_jobs=1):
    param_ranges = {
        'rsi1_period': [14, 21, 30],
        'rsi2_period': [21, 34, 45],
        'sf1': [5],
        'sf2': [5],
        'qqe1': [1.618],
        'qqe2': [5],
        'bb_length': [50],
        'bb_mult': [1],
        'trade_size': [0.01],
        'atr_multiplier': [5],
        'atr_period': [14],
    }

    keys, values = zip(*param_ranges.items())
    param_combinations = list(product(*values))
    total_combinations = len(param_combinations)

    min_data_length = max(
        max(param_ranges['rsi1_period']),
        max(param_ranges['rsi2_period']),
        max(param_ranges['sf1']),
        max(param_ranges['sf2'])
    ) + 1

    chunk_size_days = min_data_length * 2
    current_start_date = pd.to_datetime(start_date)
    all_results = []

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    while current_start_date < pd.to_datetime(end_date):
        current_end_date = current_start_date + timedelta(days=chunk_size_days)
        if current_end_date > pd.to_datetime(end_date):
            current_end_date = pd.to_datetime(end_date)

        data_chunk = data.loc[current_start_date:current_end_date]

        if len(data_chunk) < min_data_length:
            current_start_date = current_end_date + timedelta(days=1)
            continue

        param_chunks = list(chunked(param_combinations, param_chunk_size))

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for chunk in param_chunks:
                for strategy_params in chunk:
                    all_params = dict(zip(keys, strategy_params))
                    all_params['data_chunk'] = data_chunk
                    futures.append(executor.submit(run_backtest_process, **all_params))

            for future in futures:
                result = future.result()
                all_results.append(result)

        current_start_date = current_end_date + timedelta(days=1)
        if current_start_date >= pd.to_datetime(end_date):
            break

    top_results = sorted(all_results, key=lambda x: x['net_profit'], reverse=True)[:10]

    return top_results

def calculate_best_strategy_params(results_data):
    for result in results_data:
        result.setdefault('net_profit', 0)
        result.setdefault('percent_profitable', 0)
        result.setdefault('max_drawdown', 0)
        result.setdefault('total_close_trade', 0)
        result.setdefault('won_trades', 0)
        result.setdefault('percent_won', 0)

    df = pd.DataFrame(results_data)

    if df.empty:
        return pd.DataFrame(
            columns=['Net Profit', 'Total Trades', 'Max Drawdown', 'Percent Profitable', 'Won Trades','percent_won', 'strategy_params'])

    best_strategy_params = df.nlargest(10, 'net_profit')

    best_strategy_params_list = best_strategy_params.to_dict(orient='records')
    best_strategy_params_df = pd.DataFrame({
        'Net Profit': [result['net_profit'] for result in best_strategy_params_list],
        'Total Trades': [result['total_close_trade'] for result in best_strategy_params_list],
        'Max Drawdown': [result['max_drawdown'] for result in best_strategy_params_list],
        'Percent Profitable': [result['percent_profitable'] for result in best_strategy_params_list],
        'Won Trades': [result['won_trades'] for result in best_strategy_params_list],
        'percent_won': [result['percent_won'] for result in best_strategy_params_list],
        'strategy_params': [result['strategy_params'] for result in best_strategy_params_list],
    })

    return best_strategy_params_df

def save_best_strategy_params(best_strategy_params, filepath):
    for param in best_strategy_params.itertuples(index=False):
        if isinstance(param.strategy_params.get('start_date'), datetime):
            param.strategy_params['start_date'] = param.strategy_params['start_date'].strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(param.strategy_params.get('end_date'), datetime):
            param.strategy_params['end_date'] = param.strategy_params['end_date'].strftime('%Y-%m-%d %H:%M:%S')

    best_strategy_params_list = best_strategy_params.to_dict(orient='records')
    with open(filepath, 'w') as f:
        json.dump(best_strategy_params_list, f)

if __name__ == '__main__':
    file_path = r"E:\binance\pythonProject001\binanceProject01\hisdata\historydata20220101-20240728.csv"
    start_date = '2023-10-27'
    end_date = '2023-12-10'

    with ThreadPoolExecutor() as executor:
        future_data = executor.submit(load_data, file_path)
        data = future_data.result()

    optimized_results = optimize_strategy_params(data, start_date, end_date)

    if optimized_results:
        best_strategy_params = calculate_best_strategy_params(optimized_results)
        with ThreadPoolExecutor() as executor:
            executor.submit(save_best_strategy_params, best_strategy_params, 'E:/binance/pythonProject001/binanceProject01/mycode/data_in_use/best_strategy_params.json')
        df = pd.DataFrame(best_strategy_params, columns=['Net Profit', 'Total Trades', 'Max Drawdown', 'Percent Profitable', 'Won Trades', 'strategy_params'])
        df.to_csv('E:/binance/pythonProject001/binanceProject01/mycode/data_in_use/backtest_results.csv', index=False)
