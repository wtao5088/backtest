import os
import json
import time
from itertools import product
from backtrader import indicators
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import backtrader as bt
from datetime import datetime, timedelta
import pandas as pd
import dask.dataframe as dd
import sys
import threading
import numpy as np
from numpy import multiply

# 配置全局日志系统
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomStdDev(bt.Indicator):
    lines = ('stddev',)
    params = (('period', 20),)

    def __init__(CustomStdDev):
        CustomStdDev.addminperiod(CustomStdDev.params)

    def next(self):
        diff = self.data0 - self.data1
        mean = sum(diff.get(size=self.params.period)) / self.params.period
        variance = sum((x - mean) ** 2 for x in diff.get(size=self.params.period)) / self.params.period
        self.lines.stddev[0] = variance ** 0.5


# noinspection PyArgumentList
class QQEMODStrategy(bt.Strategy):
    sf2: object
    params = {
        'rsi1_period': 21,
        'rsi2_period': 34,
        'sf1': 5,
        'sf2': 5,
        'qqe1': 1.618,
        'qqe2': 5.618,
        'bb_length': 15,
        'bb_mult': 1,
        'trade_size': 0.01,
        'atr_multiplier': 5,
        'atr_period': 14,
    }

    def __init__(self):
        logger.info(f"dir(self.datas[0]): {dir(self.datas[0])}, self.datas[0].__dict__: {self.datas[0].__dict__}")

        # 数据引用
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        # 初始化列表和变量
        self.close_prices = []
        self.high_prices = []
        self.low_prices = []
        self.trend1 = 0
        self.trend2 = 0

        self.newshortband1 = []
        self.newlongband1 = []
        self.newshortband2 = []
        self.newlongband2 = []

        self.longband1 = None
        self.shortband1 = None
        self.longband2 = None
        self.shortband2 = None
        self.rsi1_period = self.params.rsi1_period
        self.rsi2_period = self.params.rsi2_period
        self.sf1 = self.params.sf1
        self.sf2 = self.params.sf2

        # RSI 指标
        self.rsi1 = bt.indicators.RSI_SMA(self.dataclose, period=self.rsi1_period)
        self.rsi2 = bt.indicators.RSI_SMA(self.dataclose, period=self.rsi2_period)

        # EMA 指标
        self.ema_rsi1 = bt.indicators.ExponentialMovingAverage(self.rsi1, period=self.sf1)
        self.ema_rsi2 = bt.indicators.ExponentialMovingAverage(self.rsi2, period=self.sf2)

        # 初始化列表
        self.dar1 = []
        self.dar2 = []
        self.atr1 = []
        self.atr2 = []

        self.cross_11 = []
        self.cross_12 = []

        self.order = None
        self.buy_signal = False
        self.sell_signal = False

        # Fast 计算
        self.fast1 = 0
        self.fast2 = 0

        # 计算 fast11 和 fast21
        self.fast11 = abs(self.fast1 - 50)
        self.fast21 = abs(self.fast2 - 50)

        # Bollinger Bands
        self.bb_length = self.params.bb_length
        self.bb_mult = self.params.bb_mult

        # 基于 fast11 和 fast21 计算 SMA
        self.basic1 = bt.indicators.SmoothedMovingAverage(self.fast11, period=self.bb_length)
        self.basic2 = bt.indicators.SmoothedMovingAverage(self.fast21, period=self.bb_length)

        # 标准差
        self.dev1 = bt.indicators.StandardDeviation(self.fast11, period=self.basic1)
        self.dev2 = bt.indicators.StandardDeviation(self.fast21, period=self.basic2)

        # 上下轨
        self.upper1 = self.basic1 + self.dev1 * self.bb_mult
        self.lower1 = self.basic1 - self.dev1 * self.bb_mult

        self.upper2 = self.basic2 + self.dev2 * self.bb_mult
        self.lower2 = self.basic2 - self.dev2 * self.bb_mult

        self.greenbar1 = 0
        self.greenbar2 = 0
        self.redbar1 = 0
        self.redbar2 = 0

        print(f"self.redbar2: {self.redbar2}")

        # ATR 和止损计算
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.params.atr_period)
        self.atr.addminperiod(self.params.atr_period)

    def log(self, txt, dt=None):
        """日志记录函数"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} - {txt}')

    def next(self):
        current_date = self.datas[0].datetime.datetime(0)
        logger.debug(f"当前日期和时间: {current_date}")

        # 更新价格数据
        current_close = self.dataclose[0]
        self.close_prices.append(current_close)
        self.high_prices.append(self.datahigh[0])
        self.low_prices.append(self.datalow[0])

        required_data_length = max(
            self.rsi1_period,
            self.rsi2_period,
            self.sf1,
            self.sf2
        ) + 1
        logger.debug(f"required_data_length: {required_data_length}")

        # 确保有足够的数据点
        if len(self.close_prices) < required_data_length:
            logger.error("数据不足，无法进行计算")
            return

        # 计算 RSI 和 EMA RSI
        rsi1_value = self.rsi1[0]
        rsi2_value = self.rsi2[0]
        ema_rsi1_value = self.ema_rsi1[0]
        ema_rsi2_value = self.ema_rsi2[0]

        # 计算 ATR 和 DAR
        if len(self.ema_rsi1) >= 2:
            atr1 = abs(self.ema_rsi1[-2] - self.ema_rsi1[-1])
            self.atr1.append(atr1)

        if len(self.ema_rsi2) >= 2:
            atr2 = abs(self.ema_rsi2[-2] - self.ema_rsi2[-1])
            self.atr2.append(atr2)

        if len(self.atr1) < 2 or len(self.atr2) < 2:
            return

        dar1 = self.atr1[-1] * self.qqe1
        dar2 = self.atr2[-1] * self.qqe2
        self.dar1.append(dar1)
        self.dar2.append(dar2)
        logger.debug(f"dar1: {dar1}, dar2: {dar2}")

        # 计算新带
        newshortband1 = ema_rsi1_value + dar1
        newlongband1 = ema_rsi1_value - dar1
        newshortband2 = ema_rsi2_value + dar2
        newlongband2 = ema_rsi2_value - dar2

        self.newshortband1.append(newshortband1)
        self.newlongband1.append(newlongband1)
        self.newshortband2.append(newshortband2)
        self.newlongband2.append(newlongband2)

        # 更新带
        # self._update_bands(self.longband1, newlongband1)
        # self._update_bands(self.shortband1, newshortband1)
        # self._update_bands(self.longband2, newlongband2)
        # self._update_bands(self.shortband2, newshortband2)

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
        print(f"newshortband1.2: {newshortband1}, newshortband2: {newshortband2}")

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
        logger.info(f"self.shortband1[0]: {self.shortband1[0]}, self.shortband2[0]: {self.shortband2[0]}")
        logger.info(f"self.longband1[0]: {self.longband1[0]}, self.longband2[0]: {self.longband2[0]}")


        if len(self.shortband1) < 3 or len(self.shortband2) < 3:
            return

        # 更新趋势和信号
        # self._update_trend_and_signals()

        if len(self.shortband1) < 3 or len(self.shortband2) < 3:
            return

        # logger.info(f"self1.ema_rsi1[-1]: {self.ema_rsi1[-1]}, self1.ema_rsi2[-1]: {self.ema_rsi2[-1]}")

        if self.rsi1[-1] <= self.shortband1[-2] and self.rsi1[-1] > self.shortband1[-1]:
            logger.info(f"RSI1 crossed above shortband1")
            self.trend1 = 1
            new_value = self.longband1[0]
            self.fast1 = []
            self.fast1.append(new_value)
            # print(f"self.fast1[0]: {self.fast1[0]}")

        elif self.rsi1[-1] >= self.shortband1[-2] and self.rsi1[-1] < self.shortband1[-1]:
            self.trend1 = -1
            new_value = self.shortband1[0]
            self.fast1 = []
            self.fast1.append(new_value)
        else:
            self.trend1 = getattr(self, 'trend1', 0)
            new_value = self.shortband1[0]
            self.fast1.append(new_value)

            # print("Trend1=", self.trend1)
            logger.debug(f"self.fast1: {self.fast1[0]}")

        if self.rsi2[-1] <= self.shortband2[-2] and self.rsi2[-1] > self.shortband2[-1]:
            logger.info(f"RSI2 crossed above shortband2")
            self.trend2 = 1
            new_value = self.longband2[0]
            self.fast2 = []
            self.fast2.append(new_value)

        elif self.rsi2[-1] >= self.shortband2[-2] and self.rsi2[-1] < self.shortband2[-1]:
            self.trend2 = -1
            new_value = self.shortband2[0]
            self.fast2 = []
            self.fast2.append(new_value)
        else:
            new_value = self.shortband2[0]
            self.fast2 = []
            self.fast2.append(new_value)

        logger.debug(f"Trend1: {self.trend1},Trend2: {self.trend2}")
        logger.debug(f"self.fast1: {self.fast1[0]},self.fast2: {self.fast2[0]}")
        logger.debug(f"当前日期和时间1: {current_date}")
        logger.debug(f"len(self.fast1): {len(self.fast1)},len(self.fast2): {len(self.fast2)}")

        fast1_value = self.fast1[0]
        fast2_value = self.fast2[0]
        fast11_value = abs(int(fast1_value - 50))
        fast21_value = abs(int(fast2_value - 50))
        self.fast11.append(fast11_value)
        self.fast21.append(fast21_value)

        logger.debug(f"self.fast11: {self.fast11[0]},self.fast21: {self.fast21[0]}")
        logger.debug(f"len(self.fast11): {len(self.fast11)},len(self.fast21): {len(self.fast21)}")

        if len(self.fast11) < 3 or len(self.fast21) < 3:
            return

        # 计算基本和标准差
        if len(self.basic1) > 0 and len(self.basic2) > 0:
            basic1_value = self.basic1[-1]
            basic2_value = self.basic2[-1]
            # self.basic11.append(abs(int(basic1_value)))
            # self.basic21.append(abs(int(basic2_value)))

        if len(self.basic11) < 2 or len(self.basic21) < 2:
            return

        stddev1_value = CustomStdDev(self.fast11, period=max(int(self.basic11[-1]), 1)).lines.stddev[0]
        stddev2_value = CustomStdDev(self.fast21, period=max(int(self.basic21[-1]), 1)).lines.stddev[0]

        # 计算上轨和下轨
        upper1 = self.basic11[-1] + abs(int(stddev1_value))
        upper2 = self.basic21[-1] + abs(int(stddev2_value))
        lower1 = self.basic11[-1] - abs(int(stddev1_value))
        lower2 = self.basic21[-1] - abs(int(stddev2_value))

        logger.debug(f"upper1: {upper1},upper2: {upper2}")
        logger.debug(f"lower1): {lower1},lower2: {lower2}")

        # 记录计算结果
        self.upper1.append(upper1)
        self.upper2.append(upper2)
        self.lower1.append(lower1)
        self.lower2.append(lower2)

        if len(self.upper1) < 2 or len(self.lower1) < 2:
            return

        # 更新买卖信号
        # self._update_signals(ema_rsi1_value, upper1, lower1, ema_rsi2_value, upper2, lower2)

    # def _update_bands(self, band, new_value):
    #     if not band:
    #         band.append(new_value)
    #     else:
    #         if self.rsi1[-1] > band[-1]:
    #             new_value = max(band[-1], new_value)
    #         band.append(new_value)

    def _update_trend_and_signals(self):
        if self.rsi1[-1] <= self.shortband1[-2] and self.rsi1[-1] > self.shortband1[-1]:
            self.trend1 = 1
            self.fast1.append(self.longband1[-1])
        elif self.rsi1[-1] >= self.shortband1[-2] and self.rsi1[-1] < self.shortband1[-1]:
            self.trend1 = -1
            self.fast1.append(self.shortband1[-1])
        else:
            self.trend1 = getattr(self, 'trend1', 0)
            self.fast1.append(self.shortband1[-1])

        if self.rsi2[-1] <= self.shortband2[-2] and self.rsi2[-1] > self.shortband2[-1]:
            self.trend2 = 1
            self.fast2.append(self.longband2[-1])
        elif self.rsi2[-1] >= self.shortband2[-2] and self.rsi2[-1] < self.shortband2[-1]:
            self.trend2 = -1
            self.fast2.append(self.shortband2[-1])
        else:
            self.fast2.append(self.shortband2[-1])

    def _update_signals(self, ema_rsi1_value, upper1, lower1, ema_rsi2_value, upper2, lower2):
        if (ema_rsi1_value - 50) > upper1:
            self.greenbar1 = 1
        else:
            self.greenbar1 = 0

        if (ema_rsi2_value - 50) > upper2:
            self.greenbar2 = 1
        else:
            self.greenbar2 = 0

        if (ema_rsi1_value - 50) < lower1:
            self.redbar1 = 1
        else:
            self.redbar1 = 0

        if (ema_rsi2_value - 50) < lower2:
            self.redbar2 = 1
        else:
            self.redbar2 = 0

        buy_signal = self.trend1 == 1 and self.trend2 == 1 and self.greenbar1 == 1 and self.greenbar2 == 1
        sell_signal = self.trend1 == -1 and self.trend2 == -1 and self.redbar1 == 1 and self.redbar2 == 1

        if buy_signal:
            if self.position.size < 0:
                self.close()
            self.buy(size=self.params['trade_size'])

        elif sell_signal:
            if self.position.size > 0:
                self.close()
            self.sell(size=self.params['trade_size'])

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('Operation profit, Total %.2f, Net %.2f' % (trade.pnl, trade.pnlcomm))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"Buy order executed, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}")
            elif order.issell():
                self.log(f"Sell order executed, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}")

def chunked(iterable, size):
    iterable = list(iterable)
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def load_data(file_path):
    logger.debug(f"加载数据: {file_path}")
    try:
        data = dd.read_csv(file_path, parse_dates=['datetime'])
        data = data.set_index('datetime')
        data = data.compute()
        logger.info(f"数据加载成功: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        return None

def run_backtest_process(data_chunk, **params):
    logger.debug(f"Received strategy_params: {params}")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(QQEMODStrategy, **params)

    data = bt.feeds.PandasData(dataname=data_chunk)
    cerebro.adddata(data)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.05)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')

    start_time = time.time()
    strat = cerebro.run()
    end_time = time.time()

    result = strat[0].analyzers.TradeAnalyzer.get_analysis()
    logger.debug(f"Trade Analyzer: {result}")
    drawdown_data = strat[0].analyzers.DrawDown.get_analysis()
    logger.debug(f"Drawdown Analysis: {drawdown_data}")
    drawdown = drawdown_data.max.drawdown if 'max' in drawdown_data else None
    net_profits = [trade.pnl.net for trade in result.get('trades', [])]
    total_net_profit = sum(net_profits)
    percent_profitable = (total_net_profit / 10000) * 100 if 10000 else 0
    total_closed_trades = result.total.closed if 'total' in result and 'closed' in result.total else 0
    won_trades = result.won.total if 'won' in result and 'total' in result.won else 0
    percent_won = (won_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0

    return {
        'net_profit': total_net_profit,
        'percent_profitable': percent_profitable,
        'max_drawdown': drawdown,
        'total_close_trade': total_closed_trades,
        'won_trades': won_trades,
        'strategy_params': params
    }

def optimize_params(data, start_date, end_date, param_chunk_size=60, n_jobs=10):
    param_ranges = {
        'rsi1_period': [14, 21],
        'rsi2_period': [14, 21, 34],
        # 'rsi1_period': list(range(5, 51, 1)),
        # 'rsi2_period': list(range(5, 51, 1)),
        'sf1': [7, 14],
        'sf2': [7, 14],
        'qqe1': [1.618],
        # 'qqe1': [1.618, 0.618, 4.236],
        'qqe2': [5],
        'bb_length': [15],
        'bb_mult': [0.05],
        # 'bb_mult': [0.05, 0.1],
        'trade_size': [0.01],
        'atr_multiplier': [5],
        'atr_period': [14],
    }

    keys, values = zip(*param_ranges.items())
    param_combinations = list(product(*values))
    total_combinations = len(param_combinations)
    logger.info(f"总参数组合数量: {total_combinations}")

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

        logger.info(f"处理数据块，从 {current_start_date} 到 {current_end_date}")
        data_chunk = data.loc[current_start_date:current_end_date]
        logger.info(f"数据块大小: {len(data_chunk)}")

        if len(data_chunk) < min_data_length:
            logger.warning(f"时间段从 {current_start_date} 到 {current_end_date} 的数据块太小，跳过")
            current_start_date = current_end_date + timedelta(days=1)
            continue

        param_chunks = list(chunked(param_combinations, param_chunk_size))

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for chunk in param_chunks:
                for params in chunk:
                    future = executor.submit(run_backtest_process, data_chunk,
                                             **{key: value for key, value in zip(keys, params)})
                    futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                except KeyError as e:
                    logger.error(f"缺少参数键: {e.args[0]}")
                except Exception as e:
                    logger.error(f"计算指标出错: {e}")

        current_start_date = current_end_date + timedelta(days=1)
        if current_start_date >= pd.to_datetime(end_date):
            logger.info("已到达结束日期，结束循环。")
            break

    top_results = sorted(all_results, key=lambda x: x['net_profit'], reverse=True)[:10]
    logger.info("已到达结束日期，结束循环。")
    return top_results

def calculate_best_params(data):
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=['Net Profit', 'Total Trades', 'Max Drawdown', 'Percent Profitable', 'Won Trades', 'params'])

    return df.nlargest(10, 'net_profit')

def save_best_params(best_params, filepath):
    with open(filepath, 'w') as f:
        json.dump(best_params.to_dict(orient='records'), f)
    logger.info("最佳参数已保存。")

def task(data):
    try:
        result = calculate_best_params(data)
        return result
    except Exception as e:
        logger.error(f"Error in process: {e}")
        raise

if __name__ == '__main__':
    file_path = r"E:\binance\pythonProject001\binanceProject01\hisdata\historydata20220101-20240728.csv"
    if not os.path.exists(file_path):
        logger.error(f"文件路径不存在: {file_path}")
        sys.exit(1)  # 退出程序并返回一个非零的退出状态
    else:
        logger.debug(f"打开文件: {file_path}")
        start_date = '2023-10-27'
        end_date = '2023-12-10'

    with ThreadPoolExecutor() as executor:
        future_data = executor.submit(load_data, file_path)
        data = future_data.result()

    if data is None:
        logger.error("数据加载失败")
        sys.exit(1)  # 退出程序并返回一个非零的退出状态

    logger.info("数据加载成功:")

    try:
        optimized_results = optimize_params(data, start_date, end_date)
    except NameError as e:
        logger.error(f"未捕获的NameError: {e}")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        raise

    if optimized_results:
        best_params = calculate_best_params(optimized_results)
        logger.info("前10名最佳参数:")
        logger.info(best_params)
        with ThreadPoolExecutor() as executor:
            executor.submit(save_best_params, best_params, 'E:/binance/pythonProject001/binanceProject01/mycode/data_in_use/best_strategy_params.json')
        df = pd.DataFrame(best_params, columns=['Net Profit', 'Total Trades', 'Max Drawdown', 'Percent Profitable', 'Won Trades', 'strategy_params'])
        df.to_csv('E:/binance/pythonProject001/binanceProject01/mycode/data_in_use/backtest_results.csv', index=False)