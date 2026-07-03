"""
训练期 vs OOT 择时质量对比
"""
import numpy as np
import akshare as ak
import pandas as pd
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'weights'
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)

def discretize_state(pct):
    if pct < BIN_EDGES[0]: return 0
    for i in range(len(BIN_EDGES)-1):
        if BIN_EDGES[i] <= pct < BIN_EDGES[i+1]: return i+1
    return len(BIN_EDGES)

def backtest_actions(Q, closes):
    day = 20
    state = discretize_state((closes[day]-closes[day-1])/closes[day-1])
    balance, shares = 10_000_000, 0
    actions = []
    for idx in range(day, len(closes)-1):
        action = int(np.argmax(Q[state]))
        actions.append(action)
        close = closes[idx]
        if action == 0:
            cb = int(balance // (close * 1000))
            if cb > 0: balance -= cb * 1000 * close; shares += cb * 1000
        elif action == 1:
            if shares > 0:
                s = min(shares, 1000 * 100)
                balance += s * close; shares -= s
        next_ret = (closes[idx+1]-closes[idx])/closes[idx]
        state = discretize_state(next_ret)
    final_value = balance + shares * closes[-1]
    total_ret = (final_value/10_000_000 - 1)*100
    bh_ret = (closes[-1]/closes[0]-1)*100
    return actions, total_ret, bh_ret

def get_data(symbol, start, end, is_index):
    if is_index:
        df = ak.stock_zh_index_daily(symbol=symbol)
    else:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust='')
    df = df.sort_values('date').reset_index(drop=True)
    mask = (pd.to_datetime(df['date']) >= pd.Timestamp(start)) & (pd.to_datetime(df['date']) <= pd.Timestamp(end))
    df = df[mask].reset_index(drop=True)
    return df['close'].values, df['date'].values

def timing_quality(actions, closes):
    correct_buy = correct_sell = 0
    total_buy = total_sell = 0
    buy_pnl = []
    sell_pnl = []

    for i, a in enumerate(actions):
        idx = i + 20
        if idx >= len(closes) - 1: break
        tomorrow_ret = (closes[idx+1] - closes[idx]) / closes[idx] * 100
        if a == 0:
            total_buy += 1
            if tomorrow_ret > 0: correct_buy += 1
            buy_pnl.append(tomorrow_ret)
        elif a == 1:
            total_sell += 1
            if tomorrow_ret < 0: correct_sell += 1
            sell_pnl.append(tomorrow_ret)

    return {
        'buy_accuracy': correct_buy/total_buy*100 if total_buy else 0,
        'sell_accuracy': correct_sell/total_sell*100 if total_sell else 0,
        'buy_trades': total_buy,
        'sell_trades': total_sell,
        'buy_avg_pnl': np.mean(buy_pnl) if buy_pnl else 0,
        'sell_avg_pnl': np.mean(sell_pnl) if sell_pnl else 0,
    }

def print_comparison(name, Q_path, train_start, train_end, test_start, test_end, is_index):
    Q = np.load(Q_path)

    rows = []
    for label, s, e in [("训练期", train_start, train_end), ("OOT", test_start, test_end)]:
        sym = 'sh600206' if not is_index else 'sh000001'
        closes, dates = get_data(sym, s, e, is_index)
        actions, ret, bh = backtest_actions(Q, closes)
        tq = timing_quality(actions, closes)
        plot_dates = dates[20:len(closes)-1]
        years = sorted(set(pd.Timestamp(d).year for d in plot_dates))

        rows.append({
            'period': label,
            'days': len(actions),
            'years': f"{years[0]}-{years[-1]}",
            'strategy': ret, 'buy_hold': bh, 'excess': ret-bh,
            'buy_n': actions.count(0), 'sell_n': actions.count(1), 'hold_n': actions.count(2),
            'buy_acc': tq['buy_accuracy'], 'sell_acc': tq['sell_accuracy'],
            'buy_avg': tq['buy_avg_pnl'], 'sell_avg': tq['sell_avg_pnl'],
        })

    tr = rows[0]; oot = rows[1]

    print(f"\n{'='*90}")
    print(f"  {name}")
    print(f"{'='*90}")
    h = f"  {'期':>5s}  {'年份':>8s}  {'天数':>5s}  {'买入':>7s}  {'卖出':>7s}  {'持有':>7s}  "
    h += f"{'策略':>9s}  {'买持':>9s}  {'超额':>9s}  {'买入准':>6s}  {'卖出准':>6s}  {'买入均':>8s}  {'卖出均':>8s}"
    print(h)
    print(f"  {'—'*90}")
    for r in rows:
        l = f"  {r['period']:>5s}  {r['years']:>8s}  {r['days']:>5d}  "
        l += f"{r['buy_n']:>4d}({r['buy_n']/r['days']*100:>3.0f}%)  "
        l += f"{r['sell_n']:>4d}({r['sell_n']/r['days']*100:>3.0f}%)  "
        l += f"{r['hold_n']:>4d}({r['hold_n']/r['days']*100:>3.0f}%)  "
        l += f"{r['strategy']:>+8.2f}%  {r['buy_hold']:>+8.2f}%  {r['excess']:>+8.2f}%  "
        l += f"{r['buy_acc']:>5.0f}%  {r['sell_acc']:>5.0f}%  "
        l += f"{r['buy_avg']:>+7.2f}%  {r['sell_avg']:>+7.2f}%"
        print(l)

    print(f"\n  【解读】")
    print(f"  训练期市场: 买持{tr['buy_hold']:+.2f}%，策略{tr['strategy']:+.2f}%，{tr['excess']:+.2f}%")
    print(f"  OOT  市场: 买持{oot['buy_hold']:+.2f}%，策略{oot['strategy']:+.2f}%，{oot['excess']:+.2f}%")

    bd = oot['buy_acc'] - tr['buy_acc']
    sd = oot['sell_acc'] - tr['sell_acc']
    print(f"  买入准: 训练{tr['buy_acc']:.0f}% → OOT{oot['buy_acc']:.0f}% ({bd:+.0f}pp)")
    print(f"  卖出准: 训练{tr['sell_acc']:.0f}% → OOT{oot['sell_acc']:.0f}% ({sd:+.0f}pp)")
    print(f"  买入均: 训练{tr['buy_avg']:+.2f}% → OOT{oot['buy_avg']:+.2f}% ({oot['buy_avg']-tr['buy_avg']:+.2f}pp)")
    print(f"  卖出均: 训练{tr['sell_avg']:+.2f}% → OOT{oot['sell_avg']:+.2f}% ({oot['sell_avg']-tr['sell_avg']:+.2f}pp)")

    # 市场状态标签
    if tr['buy_hold'] > 20:
        tr_mkt = "强牛市"
    elif tr['buy_hold'] > 5:
        tr_mkt = "弱牛市"
    elif tr['buy_hold'] > -5:
        tr_mkt = "震荡市"
    else:
        tr_mkt = "熊市"

    if oot['buy_hold'] > 20:
        oot_mkt = "强牛市"
    elif oot['buy_hold'] > 5:
        oot_mkt = "弱牛市"
    elif oot['buy_hold'] > -5:
        oot_mkt = "震荡市"
    else:
        oot_mkt = "熊市"

    print(f"  训练期市场状态: {tr_mkt} | OOT 市场状态: {oot_mkt}")
    print(f"  Q-table 在 {oot_mkt} 中的表现: {'✅ 泛化' if oot['excess'] > 0 else '❌ 失效'}")


print_comparison(
    "上证指数", WEIGHTS_DIR/"Q_table_index_best.npy",
    "20220623", "20260703",
    "20180101", "20220622", True
)

print_comparison(
    "有研新材", WEIGHTS_DIR/"Q_table_youyan_best.npy",
    "20220104", "20260703",
    "20180101", "20220103", False
)
