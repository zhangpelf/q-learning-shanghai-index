#!/usr/bin/env python3
"""
每日 Q-Learning 联合信号生成器
加载训练好的最优权重，输出今日买卖信号
"""
import numpy as np
from pathlib import Path

# ── 路径 ──
BASE = Path(__file__).parent
WEIGHTS = BASE / 'weights'

# ── 配置 ──
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)
N_ACTIONS = 3
ACTION_ICONS = {0: '✅ 买入', 1: '❌ 卖出', 2: '⏸ 持有'}


def _discretize(pct):
    if pct < BIN_EDGES[0]: return 0
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= pct < BIN_EDGES[i + 1]: return i + 1
    return len(BIN_EDGES)


def _label(s):
    if s == 0: return "<-5%"
    elif s == len(BIN_EDGES): return ">=5%"
    else: return f"[{BIN_EDGES[s - 1] * 100:.0f}%,{BIN_EDGES[s] * 100:.0f}%)"


def _signal(name, Q, closes, dates):
    r = (closes[-1] - closes[-2]) / closes[-2]
    s = _discretize(r)
    a = np.argmax(Q[s])
    return {'name': name, 'date': dates[-1], 'ret': r, 'state': s,
            'label': _label(s), 'action': a, 'icon': ACTION_ICONS[a],
            'q': Q[s], 'price': closes[-1]}


def main():
    # lazy imports 避免 circular import (pyarrow / akshare)
    import pandas as pd
    import akshare as ak

    print("=" * 60)
    print("  Q-Learning 每日联合信号")
    print("=" * 60)

    # ── 上证指数 ──
    print("\n【上证指数 sh000001】")
    df_idx = ak.stock_zh_index_daily(symbol='sh000001')
    df_idx = df_idx.sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp('20220623')
    df_idx = df_idx[pd.to_datetime(df_idx['date']) >= cutoff].reset_index(drop=True)
    closes_idx = df_idx['close'].values
    dates_idx = df_idx['date'].astype(str).values
    Q_idx = np.load(str(WEIGHTS / 'Q_table_index_best.npy'))
    sig_idx = _signal('上证指数', Q_idx, closes_idx, dates_idx)

    # ── 有研新材 ──
    print("【有研新材 600206】")
    df_stk = ak.stock_zh_a_daily(symbol='sh600206', start_date='20220101',
                                 end_date='20260703', adjust='')
    df_stk = df_stk.sort_values('date').reset_index(drop=True)
    closes_stk = df_stk['close'].values
    dates_stk = df_stk['date'].astype(str).values
    Q_stk = np.load(str(WEIGHTS / 'Q_table_youyan_best.npy'))
    sig_stk = _signal('有研新材', Q_stk, closes_stk, dates_stk)

    a_stk, a_idx = sig_stk['action'], sig_idx['action']
    print(f"\n{'=' * 60}")
    print(f"  📅 策略信号 — {sig_stk['date']}")
    print(f"{'=' * 60}")
    for sig in [sig_idx, sig_stk]:
        q = sig['q']
        print(f"\n  {sig['name']}  {sig['icon']}")
        print(f"  收盘 {sig['price']:.2f}  | {sig['ret'] * 100:+.2f}% → S{sig['state']} {sig['label']}")
        print(f"  Q: 买入={q[0]:+.2f}  卖出={q[1]:+.2f}  持有={q[2]:+.2f}")
    print(f"\n  → ", end='')
    if a_stk == 0 and a_idx == 0: print("两市一致买入，积极做多")
    elif a_stk == 1 and a_idx == 1: print("两市一致卖出，规避风险")
    elif a_stk == 2 and a_idx == 2: print("两市一致持有，持仓观望")
    elif a_stk == 0 and a_idx == 2: print("个股看多 / 指数中性，精选个股")
    elif a_stk == 2 and a_idx == 0: print("指数看多 / 个股中性，侧重指数")
    elif a_stk == 1: print("个股看空，注意仓位")
    else: print("信号分歧，谨慎操作")


if __name__ == '__main__':
    main()
