"""
OOT 可视化：价格走势 + 买卖持有标注
"""
import numpy as np
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'weights'
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)
TRADE_UNIT = 1000
BALANCE_INIT = 10_000_000

def discretize_state(pct):
    if pct < BIN_EDGES[0]: return 0
    for i in range(len(BIN_EDGES)-1):
        if BIN_EDGES[i] <= pct < BIN_EDGES[i+1]: return i+1
    return len(BIN_EDGES)

def backtest_with_actions(Q, closes, dates):
    day = 20
    state = discretize_state((closes[day]-closes[day-1])/closes[day-1])
    balance, shares = BALANCE_INIT, 0
    actions = []
    navs = []
    for idx in range(day, len(closes)-1):
        action = int(np.argmax(Q[state]))
        actions.append(action)
        close = closes[idx]
        if action == 0:
            cb = int(balance // (close * TRADE_UNIT))
            if cb > 0:
                balance -= cb * TRADE_UNIT * close
                shares += cb * TRADE_UNIT
        elif action == 1:
            if shares > 0:
                s = min(shares, TRADE_UNIT * 100)
                balance += s * close
                shares -= s
        next_ret = (closes[idx+1]-closes[idx])/closes[idx]
        state = discretize_state(next_ret)
        nav = balance + shares * closes[-1]
        navs.append(nav)
    navs.append(balance + shares * closes[-1])
    return actions, navs

# ── 获取数据 ──
def get_data(symbol, start, end, is_index=True):
    if is_index:
        df = ak.stock_zh_index_daily(symbol=symbol)
    else:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust='')
    df = df.sort_values('date').reset_index(drop=True)
    mask = (pd.to_datetime(df['date']) >= pd.Timestamp(start)) & (pd.to_datetime(df['date']) <= pd.Timestamp(end))
    df = df[mask].reset_index(drop=True)
    return df['close'].values, pd.to_datetime(df['date'].values)

# ── 画图 ──
def plot_strategy(name, symbol, Q_path, start, end, is_index, filename):
    Q = np.load(Q_path)
    closes, dates = get_data(symbol, start, end, is_index)
    actions, navs = backtest_with_actions(Q, closes, dates)

    # 对齐长度
    plot_dates = dates[20:len(closes)-1]
    plot_closes = closes[20:len(closes)-1]
    plot_navs = np.array(navs[:-1]) / BALANCE_INIT * closes[20]  # 归一化到价格尺度

    buy_idx  = [i for i, a in enumerate(actions) if a == 0]
    sell_idx = [i for i, a in enumerate(actions) if a == 1]
    hold_idx = [i for i, a in enumerate(actions) if a == 2]

    fig, ax = plt.subplots(figsize=(16, 6))

    # 价格线
    ax.plot(plot_dates, plot_closes, color='#888888', linewidth=1, alpha=0.7, label='收盘价')

    # 买卖标记
    ax.scatter(plot_dates[0::5], plot_closes[0::5], color='#888888', s=2, alpha=0.3)  # hold hint

    marker_scale = 25
    if buy_idx:
        ax.scatter(plot_dates[buy_idx], plot_closes[buy_idx],
                   color='#e74c3c', marker='^', s=marker_scale*3, alpha=0.7, label='买入', zorder=5)
    if sell_idx:
        ax.scatter(plot_dates[sell_idx], plot_closes[sell_idx],
                   color='#2ecc71', marker='v', s=marker_scale*3, alpha=0.7, label='卖出', zorder=5)
    if hold_idx:
        ax.scatter(plot_dates[hold_idx], plot_closes[hold_idx],
                   color='#3498db', marker='.', s=marker_scale, alpha=0.3, label='持有', zorder=2)

    ax.set_title(f'{name} OOT 策略动作分布 (2018~训练前)', fontsize=14, fontweight='bold')
    ax.set_ylabel('价格')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"  保存: {filename}")
    plt.close(fig)

    # 统计
    total = len(actions)
    print(f"  {name}: 买入={len(buy_idx)}({len(buy_idx)/total*100:.0f}%) "
          f"卖出={len(sell_idx)}({len(sell_idx)/total*100:.0f}%) "
          f"持有={len(hold_idx)}({len(hold_idx)/total*100:.0f}%)")

# ── 跑 ──
plot_strategy("上证指数", "sh000001",
              WEIGHTS_DIR/"Q_table_index_best.npy",
              "20180101", "20220622", True,
              WEIGHTS_DIR/"../oot_index_actions.png")

plot_strategy("有研新材", "sh600206",
              WEIGHTS_DIR/"Q_table_youyan_best.npy",
              "20180101", "20220103", False,
              WEIGHTS_DIR/"../oot_youyan_actions.png")

print("\n两张图已保存到 rl_training/ 目录")
