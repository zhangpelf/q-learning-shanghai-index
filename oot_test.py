"""
Out-Of-Time 测试：用训练好的 Q-table 在更早的数据上回测（不改权重）
指数：训练 2022-06~2026-07 → 测试 2020-01~2022-06
个股：训练 2022-01~2026-07 → 测试 2018-01~2022-01
"""
import numpy as np
import akshare as ak
import pandas as pd
from pathlib import Path

# ── 共用参数 ──
BALANCE_INIT = 10_000_000
TRADE_UNIT = 1000
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)
N_STATES = len(BIN_EDGES) + 1
N_ACTIONS = 3

WEIGHTS_DIR = Path(__file__).parent / 'weights'


def discretize_state(pct_change):
    if pct_change < BIN_EDGES[0]:
        return 0
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= pct_change < BIN_EDGES[i + 1]:
            return i + 1
    return len(BIN_EDGES)


def state_label(state):
    if state == 0:
        return "<-5%"
    elif state == len(BIN_EDGES):
        return ">=5%"
    else:
        return f"[{BIN_EDGES[state-1]*100:.0f}%,{BIN_EDGES[state]*100:.0f}%)"


def backtest(Q, closes):
    """Greedy backtest: always pick argmax(Q[state]), no training."""
    day = 20
    if day >= len(closes) - 1:
        return None

    state = discretize_state((closes[day] - closes[day - 1]) / closes[day - 1])
    balance = BALANCE_INIT
    shares = 0
    trades = {'buy': 0, 'sell': 0, 'hold': 0}
    navs = [balance]
    actions_log = []

    for _ in range(day, len(closes) - 1):
        action = int(np.argmax(Q[state]))
        actions_log.append(action)
        close = closes[day]
        trades[['buy', 'sell', 'hold'][action]] += 1

        if action == 0:  # buy
            can_buy = int(balance // (close * TRADE_UNIT))
            if can_buy > 0:
                balance -= can_buy * TRADE_UNIT * close
                shares += can_buy * TRADE_UNIT
        elif action == 1:  # sell
            if shares > 0:
                sell_shares = min(shares, TRADE_UNIT * 100)
                balance += sell_shares * close
                shares -= sell_shares

        day += 1
        if day >= len(closes):
            break
        next_ret = (closes[day] - closes[day - 1]) / closes[day - 1]
        state = discretize_state(next_ret)
        nav = balance + shares * closes[-1]  # mark-to-end for tracking
        navs.append(nav)

    final_value = balance + shares * closes[min(day, len(closes) - 1)]
    total_ret = (final_value / BALANCE_INIT - 1) * 100
    bh_ret = (closes[-1] / closes[0] - 1) * 100
    excess = total_ret - bh_ret
    return {
        'total_return': total_ret,
        'buy_hold': bh_ret,
        'excess': excess,
        'trades': trades,
        'navs': navs,
        'actions': actions_log,
    }


def run_test(name, symbol, Q_path, data_source, start, end, is_index=True):
    print(f"\n{'='*60}")
    print(f"  {name} ({symbol})")
    print(f"  数据源: {data_source}")
    print(f"  测试区间: {start} ~ {end}")
    print(f"{'='*60}")

    # 加载 Q-table
    Q = np.load(Q_path)
    print(f"  Q-table: {Q_path.name}, shape={Q.shape}")

    # 获取数据
    if is_index:
        df = ak.stock_zh_index_daily(symbol=symbol)
    else:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust='')

    df = df.sort_values('date').reset_index(drop=True)
    cutoff_start = pd.Timestamp(start)
    cutoff_end = pd.Timestamp(end)
    df = df[(pd.to_datetime(df['date']) >= cutoff_start) &
            (pd.to_datetime(df['date']) <= cutoff_end)].reset_index(drop=True)

    if len(df) < 50:
        print(f"  ❌ 数据不足: {len(df)} 天，跳过")
        return

    closes = df['close'].values
    dates = df['date'].astype(str).values
    print(f"  天数: {len(closes)}, 区间: {dates[0]} ~ {dates[-1]}")

    result = backtest(Q, closes)
    if result is None:
        print(f"  ❌ 无法回测（数据太短）")
        return

    print(f"\n  ── 结果 ──")
    print(f"  策略收益: {result['total_return']:.2f}%")
    print(f"  买持收益: {result['buy_hold']:.2f}%")
    print(f"  超额收益: {result['excess']:+.2f}%")
    print(f"  交易分布: 买入={result['trades']['buy']}, "
          f"卖出={result['trades']['sell']}, "
          f"持有={result['trades']['hold']}")
    print(f"  交易频率: {result['trades']['buy']+result['trades']['sell']}/{len(closes)-20} "
          f"({(result['trades']['buy']+result['trades']['sell'])/(len(closes)-20)*100:.0f}%)")

    # 分段分析
    n = len(closes)
    third = n // 3
    segments = [(0, third, "前1/3"), (third, 2*third, "中1/3"), (2*third, n, "后1/3")]
    print(f"\n  ── 分段表现 ──")
    for s_start, s_end, label in segments:
        seg_closes = closes[s_start:s_end]
        if len(seg_closes) < 20:
            continue
        seg_bh = (seg_closes[-1] / seg_closes[0] - 1) * 100
        print(f"    {label}: 买持 {seg_bh:+.2f}%")

    # 每个 state 的推荐动作
    print(f"\n  ── 策略画像（Q-table 每个 state 的最优动作） ──")
    action_str = ['买入', '卖出', '持有']
    for s in range(N_STATES):
        best = int(np.argmax(Q[s]))
        vals = Q[s]
        print(f"    {state_label(s):>8s} → {action_str[best]}  "
              f"(Q: 买={vals[0]:+7.2f}, 卖={vals[1]:+7.2f}, 持={vals[2]:+7.2f})")


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # ── 1. 指数 OOT ──
    run_test(
        name="上证指数",
        symbol="sh000001",
        Q_path=WEIGHTS_DIR / "Q_table_index_best.npy",
        data_source="stock_zh_index_daily",
        start="20180101",
        end="20220622",
        is_index=True,
    )

    # ── 2. 个股 OOT ──
    run_test(
        name="有研新材",
        symbol="sh600206",
        Q_path=WEIGHTS_DIR / "Q_table_youyan_best.npy",
        data_source="stock_zh_a_daily",
        start="20180101",
        end="20220103",
        is_index=False,
    )
