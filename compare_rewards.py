"""
奖励函数迭代对比 — 上证指数(sh000001)
测试 4 种不同的奖励机制，选出最优
"""
import numpy as np
import pandas as pd
import akshare as ak
from pathlib import Path

# ── 共用配置 ──
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
BIN_EDGES = np.arange(-0.05, 0.06, 0.01)
N_STATES = len(BIN_EDGES) + 1
N_ACTIONS = 3

WEIGHTS_DIR = Path(__file__).parent / 'weights'

# ── 数据 ──
def fetch_data():
    df = ak.stock_zh_index_daily(symbol=STOCK_SYMBOL)
    df = df.sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp(DATA_START)
    df = df[pd.to_datetime(df['date']) >= cutoff].reset_index(drop=True)
    closes = df['close'].values
    dates = df['date'].values
    pct_changes = np.diff(closes) / closes[:-1]
    print(f"  数据: {len(closes)} 天, {dates[0]} ~ {dates[-1]}")
    print(f"  买持收益: {(closes[-1]/closes[0]-1)*100:.2f}%")
    return closes, pct_changes, dates

def discretize_state(pct_change):
    if pct_change < BIN_EDGES[0]:
        return 0
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= pct_change < BIN_EDGES[i+1]:
            return i + 1
    return len(BIN_EDGES)

def state_label(state):
    if state == 0: return "<-5%"
    elif state == len(BIN_EDGES): return ">=5%"
    else: return f"[{BIN_EDGES[state-1]*100:.0f}%,{BIN_EDGES[state]*100:.0f}%)"

# ── 交易环境（可插拔奖励函数） ──
class TradingEnv:
    reward_name = "DEFAULT"

    def __init__(self, closes):
        self.closes = closes
        self.n_days = len(closes)

    def reset(self):
        self.day = np.random.randint(0, self.n_days - 10)
        self.balance = BALANCE_INIT
        self.shares = 0
        return self._get_state()

    def _get_state(self):
        if self.day == 0: return 0
        pct = (self.closes[self.day] - self.closes[self.day-1]) / self.closes[self.day-1]
        return discretize_state(pct)

    def _reward(self, action, tomorrow_ret):
        return 0  # override

    def step(self, action):
        close = self.closes[self.day]
        if action == 0:
            can_buy = int(self.balance // (close * TRADE_UNIT))
            if can_buy > 0:
                self.balance -= can_buy * TRADE_UNIT * close
                self.shares += can_buy * TRADE_UNIT
        elif action == 1:
            if self.shares > 0:
                sell_shares = min(self.shares, TRADE_UNIT * 100)
                self.balance += sell_shares * close
                self.shares -= sell_shares
        self.day += 1
        done = self.day >= self.n_days - 1
        if not done:
            tomorrow_ret = (self.closes[self.day+1] - self.closes[self.day]) / self.closes[self.day]
            reward = self._reward(action, tomorrow_ret)
        else:
            reward = 0
        return self._get_state(), reward, done


# ── 奖励函数 1: 方向性（去偏移，去摩擦） ──
class EnvDir(TradingEnv):
    reward_name = "方向性(去漂移)"

    def _reward(self, action, r):
        if action == 0:   # 买入 → 涨+1, 跌-1
            return 1 if r > 0 else -1
        elif action == 1: # 卖出 → 跌+1, 涨-1
            return 1 if r < 0 else -1
        else:
            return 0  # 持有 = 0 基线


# ── 奖励函数 2: 幅度的绝对值（无摩擦，持有中性） ──
class EnvAbs(TradingEnv):
    reward_name = "幅度(无摩擦)"

    def _reward(self, action, r):
        scale = 2
        if action == 0:
            return r * 100 * scale
        elif action == 1:
            return -r * 100 * scale
        else:
            return 0


# ── 奖励函数 3: 不对称（买入更激进，卖出更保守） ──
class EnvAsym(TradingEnv):
    reward_name = "不对称(买激进/卖保守)"

    def _reward(self, action, r):
        if action == 0:
            return r * 100 * 8    # 买入放大
        elif action == 1:
            return -r * 100 * 3   # 卖出缩小
        else:
            return 0


# ── 奖励函数 4: 阈值触发（只有超阈值才奖励交易） ──
class EnvTresh(TradingEnv):
    reward_name = "阈值触发(>0.5%)"

    def _reward(self, action, r):
        if action == 2: return 0
        correct = (action == 0 and r > 0) or (action == 1 and r < 0)
        wrong = (action == 0 and r < 0) or (action == 1 and r > 0)
        if abs(r) > 0.005:  # 超过 0.5% 阈值
            return 2 if correct else -2
        else:
            return 0.5 if correct else -0.5


# ── 训练 ──
def train(env_class, closes):
    env = env_class(closes)
    np.random.seed(RANDOM_SEED)
    Q = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START

    for ep in range(N_EPISODES):
        state = env.reset()
        total = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = np.argmax(Q[state])
            next_state, reward, done = env.step(action)
            total += reward
            Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
            state = next_state
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    return Q


# ── 回测 ──
def backtest(Q, env_class, closes):
    env = env_class(closes)
    env.reset()
    env.day = 20
    state = env._get_state()
    trades = {'buy': 0, 'sell': 0, 'hold': 0}

    for _ in range(env.day, env.n_days - 1):
        action = np.argmax(Q[state])
        trades[['buy', 'sell', 'hold'][action]] += 1
        next_state, reward, done = env.step(action)
        if done: break
        state = next_state

    final_value = env.balance + env.shares * closes[env.day]
    total_ret = (final_value - BALANCE_INIT) / BALANCE_INIT * 100
    buy_hold_ret = (closes[-1] - closes[0]) / closes[0] * 100
    return {'return': total_ret, 'buy_hold': buy_hold_ret, 'excess': total_ret - buy_hold_ret, 'trades': trades}


# ── 输出 Q 矩阵 ──
def print_q(Q, name):
    action_names = ['买入', '卖出', '持有']
    print(f"\n  {name}")
    for s in range(N_STATES):
        best_a = np.argmax(Q[s])
        q_str = " ".join(f"{Q[s][a]:>+7.2f}" for a in range(N_ACTIONS))
        print(f"    S{s:2d} {state_label(s):>6} | {q_str} → {action_names[best_a]}")
    counts = [sum(1 for s in range(N_STATES) if np.argmax(Q[s]) == a) for a in range(3)]
    print(f"    策略分布: 买入{counts[0]} / 卖出{counts[1]} / 持有{counts[2]}")


# ── 主流程 ──
def main():
    print("=" * 60)
    print(f"    奖励函数对比 — {STOCK_NAME}({STOCK_SYMBOL})")
    print("=" * 60)
    closes, pct_changes, dates = fetch_data()

    envs = [EnvDir, EnvAbs, EnvAsym, EnvTresh]
    results = []

    for env_class in envs:
        print(f"\n{'─'*50}")
        print(f"🔥 训练: {env_class.reward_name}")
        Q = train(env_class, closes)
        print_q(Q, env_class.reward_name)
        bt = backtest(Q, env_class, closes)
        results.append((env_class, bt))
        status = "✅" if bt['excess'] > 0 else "❌"
        print(f"  📊 策略={bt['return']:+.2f}% | 买持={bt['buy_hold']:+.2f}% | 超额={bt['excess']:+.2f}% {status}")
        print(f"     交易: 买入{bt['trades']['buy']} / 卖出{bt['trades']['sell']} / 持有{bt['trades']['hold']}")

    print(f"\n{'='*60}")
    print(f"  排名")
    print(f"{'='*60}")
    ranked = sorted(results, key=lambda x: x[1]['excess'], reverse=True)
    for i, (env_class, bt) in enumerate(ranked):
        print(f"  #{i+1} {env_class.reward_name:>14s} | 策略={bt['return']:>+7.2f}% | 超额={bt['excess']:>+7.2f}% | "
              f"买{bt['trades']['buy']}/卖{bt['trades']['sell']}/持{bt['trades']['hold']}")

    # 用最好的奖励函数重新训练并保存
    best_env, best_bt = ranked[0]
    print(f"\n{'='*60}")
    print(f"  ✅ 最优: {best_env.reward_name} — 保存权重")
    print(f"{'='*60}")
    np.random.seed(RANDOM_SEED)
    Q_best = train(best_env, closes)

    import json, os
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    np.save(WEIGHTS_DIR / 'Q_table_index_best.npy', Q_best)

    lines = [f"Q-Learning 最优奖励 — {STOCK_NAME}({STOCK_SYMBOL})",
             f"奖励函数: {best_env.reward_name}",
             f"数据范围: {dates[0]} ~ {dates[-1]}",
             f"保存时间: {pd.Timestamp.now()}",
             f"训练轮数: {N_EPISODES}, α={ALPHA}, γ={GAMMA}, seed={RANDOM_SEED}"]
    lines.append(f"\n{'状态':>6} {'涨跌幅':>6} {'买入':>8} {'卖出':>8} {'持有':>8} {'动作':>6}")
    lines.append("-"*50)
    action_names = ['买入', '卖出', '持有']
    for s in range(N_STATES):
        q_str = " ".join(f"{Q_best[s][a]:>+8.2f}" for a in range(N_ACTIONS))
        lines.append(f"{s:>6} {state_label(s):>6} {q_str} {action_names[np.argmax(Q_best[s])]:>6}")
    lines.append(f"\n回测收益: {best_bt['return']:.2f}%")
    lines.append(f"买入持有: {best_bt['buy_hold']:.2f}%")
    lines.append(f"超额收益: {best_bt['excess']:.2f}%")
    lines.append(f"交易分布: {best_bt['trades']}")

    with open(WEIGHTS_DIR / 'Q_table_index_best.txt', 'w') as f:
        f.write('\n'.join(lines))

    meta = {
        'index': STOCK_NAME, 'symbol': STOCK_SYMBOL,
        'algorithm': 'Q-learning', 'reward_function': best_env.reward_name,
        'data_range': f'{dates[0]}~{dates[-1]}',
        'n_episodes': N_EPISODES, 'alpha': ALPHA, 'gamma': GAMMA, 'seed': RANDOM_SEED,
        'backtest_return': best_bt['return'],
        'buy_hold_return': best_bt['buy_hold'],
        'excess_return': best_bt['excess'],
        'trade_distribution': best_bt['trades'],
        'saved_at': str(pd.Timestamp.now()),
    }
    with open(WEIGHTS_DIR / 'metadata_index_best.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    today_ret = (closes[-1] - closes[-2]) / closes[-2]
    state = discretize_state(today_ret)
    best_a = np.argmax(Q_best[state])
    labels = ['✅ 买入', '❌ 卖出', '⏸ 持有']
    print(f"\n📅 今日({dates[-1]})信号: {today_ret*100:+.2f}% → S{state} → {labels[best_a]}")

    print(f"\n💾 最优权重已保存: weights/Q_table_index_best.npy")

if __name__ == '__main__':
    main()
