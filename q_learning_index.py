"""
Q-Learning 交易策略 — 上证指数(sh000001)
===================
架构：沿用有研新材 RL 方案（统一模板）
- 状态: 12 bins (1% 均匀分桶, -5%~+5%)
- 动作: 0=买入, 1=卖出, 2=持有
- 奖励: 阶梯比例（按次日涨跌幅成比例）
- 训练: 1000 episodes
- 落盘: Q-table 权重保存到 weights/
"""

import numpy as np
import pandas as pd
import akshare as ak
import os
import json
from pathlib import Path

# ── 配置 ──
STOCK_SYMBOL = 'sh000001'
STOCK_NAME = '上证指数'
DATA_START = '20220623'
DATA_END = '20260703'
N_EPISODES = 1000
ALPHA = 0.01
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.998
BALANCE_INIT = 10_000_000
TRADE_UNIT = 1000
RANDOM_SEED = 42

BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / 'weights'

# 12 个状态边界（1% 均匀分桶）
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)  # [-5%, -4%, ..., +5%]
N_STATES = len(BIN_EDGES) + 1  # 12
N_ACTIONS = 3  # 0=买入, 1=卖出, 2=持有

# ── 数据获取 ──────────────────────────────────────────
def fetch_data():
    print(f"📡 获取 {STOCK_NAME}({STOCK_SYMBOL}) 数据...")
    df = ak.stock_zh_index_daily(symbol=STOCK_SYMBOL)
    df = df.sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp(DATA_START)
    df = df[pd.to_datetime(df['date']) >= cutoff].reset_index(drop=True)
    print(f"  数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    closes = df['close'].values
    dates = df['date'].values
    pct_changes = np.diff(closes) / closes[:-1]

    print(f"  数据量: {len(closes)} 天, {len(pct_changes)} 个交易日")
    print(f"  日期: {dates[0]} ~ {dates[-1]}")
    print(f"  今日({dates[-1]})收盘价: {closes[-1]:.2f}")
    print(f"  今日涨跌幅: {pct_changes[-1]*100:.2f}%")
    print(f"  日涨跌幅: min={pct_changes.min()*100:.2f}%, max={pct_changes.max()*100:.2f}%, "
          f"mean={pct_changes.mean()*100:.2f}%, std={pct_changes.std()*100:.2f}%")
    return closes, pct_changes, dates

# ── 状态离散化 ────────────────────────────────────────
def discretize_state(pct_change):
    """将涨跌幅映射到 0-11 状态"""
    if pct_change < BIN_EDGES[0]:
        return 0
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= pct_change < BIN_EDGES[i+1]:
            return i + 1
    return len(BIN_EDGES)  # >= +5%

def state_label(state):
    """状态的可读标签"""
    if state == 0:
        return "<-5%"
    elif state == len(BIN_EDGES):
        return ">=5%"
    else:
        lo = BIN_EDGES[state-1] * 100
        hi = BIN_EDGES[state] * 100
        return f"[{lo:.0f}%,{hi:.0f}%)"

# ── 交易环境 ──────────────────────────────────────────
class TradingEnv:
    def __init__(self, closes):
        self.closes = closes
        self.n_days = len(closes)

    def reset(self):
        self.day = np.random.randint(0, self.n_days - 10)
        self.balance = BALANCE_INIT
        self.shares = 0
        self.asset_values = [self.balance]
        return self._get_state()

    def _get_state(self):
        if self.day == 0:
            return 0
        pct = (self.closes[self.day] - self.closes[self.day-1]) / self.closes[self.day-1]
        return discretize_state(pct)

    def step(self, action):
        day = self.day
        close = self.closes[day]

        # 执行动作
        if action == 0:  # 买入
            can_buy = self.balance // (close * TRADE_UNIT)
            if can_buy > 0:
                cost = can_buy * TRADE_UNIT * close
                self.balance -= cost
                self.shares += can_buy * TRADE_UNIT
        elif action == 1:  # 卖出
            if self.shares > 0:
                sell_shares = min(self.shares, TRADE_UNIT * 100)
                self.balance += sell_shares * close
                self.shares -= sell_shares

        self.day += 1
        done = self.day >= self.n_days - 1

        # 阶梯奖励
        if not done:
            tomorrow_ret = (self.closes[min(self.day+1, self.n_days-1)] - self.closes[self.day]) / self.closes[self.day]
            if action == 0:      # 买入 → 涨奖跌罚
                reward = tomorrow_ret * 100 * 5
            elif action == 1:    # 卖出 → 跌奖涨罚
                reward = -tomorrow_ret * 100 * 5
            else:                # 持有 → 涨奖跌罚(温和)
                reward = tomorrow_ret * 100 * 2
            if action == 0 or action == 1:
                reward -= 1  # 交易摩擦
        else:
            reward = 0

        next_state = self._get_state()
        return next_state, reward, done

# ── Q-learning 训练 ───────────────────────────────────
def train_q_learning(closes, n_episodes=N_EPISODES):
    print(f"\n🧠 开始 {n_episodes} 轮 Q-learning 训练...")
    print(f"   参数: α={ALPHA}, γ={GAMMA}, ε={EPSILON_START}→{EPSILON_END}, seed={RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)

    Q = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(n_episodes):
        env = TradingEnv(closes)
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            total_reward += reward
            Q[state, action] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % 200 == 0:
            avg_r = np.mean(episode_rewards[-200:])
            print(f"   Episode {ep+1:5d}/{n_episodes} | ε={epsilon:.3f} | avg_R={avg_r:.1f}")

    print(f" ✅ 训练完成")
    return Q, episode_rewards

# ── 策略分析 ──────────────────────────────────────────
def analyze_policy(Q, verbose=True):
    action_names = ['买入', '卖出', '持有']
    if verbose:
        print("\n" + "="*70)
        print(f"{'📊 Q 矩阵 — 上证指数(sh000001)':^68}")
        print("="*70)
        print(f"{'状态':>8} {'涨跌幅':>10} {'买入':>8} {'卖出':>8} {'持有':>8} {'→ 动作':>8}")
        print("-"*70)

    policy = {}
    for s in range(N_STATES):
        best_a = np.argmax(Q[s])
        label = state_label(s)
        policy[label] = action_names[best_a]
        if verbose:
            q_str = " ".join(f"{Q[s][a]:>+8.2f}" for a in range(N_ACTIONS))
            print(f"{s:>8} {label:>10} {q_str} {'→':>4} {action_names[best_a]:>4}")

    if verbose:
        print("-"*70)
        buy_states = sum(1 for s in range(N_STATES) if np.argmax(Q[s]) == 0)
        sell_states = sum(1 for s in range(N_STATES) if np.argmax(Q[s]) == 1)
        hold_states = sum(1 for s in range(N_STATES) if np.argmax(Q[s]) == 2)
        print(f"   策略分布: 买入={buy_states} | 卖出={sell_states} | 持有={hold_states}")
        print("="*70)
    return policy

# ── 回测 ──────────────────────────────────────────────
def backtest(Q, closes):
    env = TradingEnv(closes)
    state = env.reset()
    env.day = 20
    state = env._get_state()

    total_trades = {'buy': 0, 'sell': 0, 'hold': 0}

    for step in range(env.day, env.n_days - 1):
        action = np.argmax(Q[state])
        if action == 0: total_trades['buy'] += 1
        elif action == 1: total_trades['sell'] += 1
        else: total_trades['hold'] += 1
        next_state, reward, done = env.step(action)
        if done: break
        state = next_state

    final_value = env.balance + env.shares * closes[env.day]
    total_return = (final_value - BALANCE_INIT) / BALANCE_INIT * 100
    buy_hold_ret = (closes[-1] - closes[0]) / closes[0] * 100

    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_ret,
        'excess_return': total_return - buy_hold_ret,
        'final_value': final_value,
        'trades': total_trades,
    }

# ── 今日判断 ──────────────────────────────────────────
def today_signal(Q, closes, dates):
    today_ret = (closes[-1] - closes[-2]) / closes[-2]
    state = discretize_state(today_ret)
    action = np.argmax(Q[state])
    action_names = ['✅ 买入', '❌ 卖出', '⏸ 持有']

    print(f"\n{'='*60}")
    print(f"📅 今日信号 — {dates[-1]} (周五)")
    print(f"{'='*60}")
    print(f"  指数: {STOCK_NAME} ({STOCK_SYMBOL})")
    print(f"  今日涨跌幅: {today_ret*100:.2f}% → 状态 {state} ({state_label(state)})")
    print(f"  Q值: 买入={Q[state][0]:+.2f}, 卖出={Q[state][1]:+.2f}, 持有={Q[state][2]:+.2f}")
    print(f"  🎯 建议动作: {action_names[action]}")
    if action == 0:
        print(f"  💡 理由是: 当前涨跌幅进入历史买入区间，模型期望回报为正")
    elif action == 1:
        print(f"  💡 理由是: 模型判断当前风险较大，建议卖出/观望")
    else:
        print(f"  💡 理由是: 模型建议不操作，等待更明确信号")
    print(f"{'='*60}\n")
    return action, state

# ── 权重保存 ──────────────────────────────────────────
def save_weights(Q, episode_rewards, policy, bt):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Q 表数值
    np.save(WEIGHTS_DIR / 'Q_table_index.npy', Q)

    # Q 表文本
    lines = [f"Q-Learning 权重 — {STOCK_NAME}({STOCK_SYMBOL})",
             f"保存时间: {pd.Timestamp.now()}",
             f"训练轮数: {N_EPISODES}, α={ALPHA}, γ={GAMMA}\n"]
    lines.append(f"{'状态':>8} {'涨跌幅':>10} {'买入':>10} {'卖出':>10} {'持有':>10} {'动作':>6}")
    lines.append("-"*60)
    action_names = ['买入', '卖出', '持有']
    for s in range(N_STATES):
        label = state_label(s)
        best_a = np.argmax(Q[s])
        q_str = " ".join(f"{Q[s][a]:>+10.2f}" for a in range(N_ACTIONS))
        lines.append(f"{s:>8} {label:>10} {q_str} {action_names[best_a]:>6}")
    lines.append(f"\n回测收益: {bt['total_return']:.2f}%")
    lines.append(f"买入持有: {bt['buy_hold_return']:.2f}%")
    lines.append(f"超额收益: {bt['excess_return']:.2f}%")
    lines.append(f"交易分布: {bt['trades']}")

    with open(WEIGHTS_DIR / 'Q_table_index.txt', 'w') as f:
        f.write('\n'.join(lines))

    np.save(WEIGHTS_DIR / 'episode_rewards_index.npy', episode_rewards)

    meta = {
        'index': STOCK_NAME, 'symbol': STOCK_SYMBOL,
        'algorithm': 'Q-learning', 'n_states': N_STATES, 'n_actions': N_ACTIONS,
        'n_episodes': N_EPISODES, 'alpha': ALPHA, 'gamma': GAMMA,
        'backtest_return': bt['total_return'],
        'buy_hold_return': bt['buy_hold_return'],
        'excess_return': bt['excess_return'],
        'trade_distribution': bt['trades'],
        'saved_at': str(pd.Timestamp.now()),
    }
    with open(WEIGHTS_DIR / 'metadata_index.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n💾 权重已保存到: {WEIGHTS_DIR}/")
    for name in ['Q_table_index.npy', 'Q_table_index.txt', 'episode_rewards_index.npy', 'metadata_index.json']:
        fpath = WEIGHTS_DIR / name
        if fpath.exists():
            print(f"   📄 {name} ({os.path.getsize(fpath):,} bytes)")

# ── 主流程 ────────────────────────────────────────────
def main():
    print("╔" + "═"*58 + "╗")
    print(f"║  Q-Learning 交易策略 — {STOCK_NAME}({STOCK_SYMBOL}){'':>22}║")
    print("╚" + "═"*58 + "╝")

    closes, pct_changes, dates = fetch_data()
    Q, rewards = train_q_learning(closes, N_EPISODES)
    policy = analyze_policy(Q)
    bt = backtest(Q, closes)

    print(f"\n📈 回测结果:")
    print(f"   策略收益: {bt['total_return']:+.2f}%")
    print(f"   买入持有: {bt['buy_hold_return']:+.2f}%")
    print(f"   超额收益: {bt['excess_return']:+.2f}%")
    print(f"   交易分布: 买入{bt['trades']['buy']}次 / 卖出{bt['trades']['sell']}次 / 持有{bt['trades']['hold']}次")

    today_action, today_state = today_signal(Q, closes, dates)
    save_weights(Q, rewards, policy, bt)

    print(f"\n🎯 大盘仓位建议:")
    if today_action == 0:
        print(f"   🔴 信号: 【买入】")
    elif today_action == 1:
        print(f"   🟢 信号: 【卖出】")
    else:
        print(f"   ⚪ 信号: 【持有】")

if __name__ == '__main__':
    main()
