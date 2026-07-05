# Q-Learning 上证指数交易策略

> 一个极简的 Q-Learning 交易策略，用 36 个参数（12 状态 × 3 动作）在指数上实现年化超额 +22%
>
> **训练数据**: 2022-06 ~ 2026-07 上证指数日线 · **训练开销**: 1000 轮 ≈ 0.3 秒（Mac CPU）

## 核心洞察

**奖励函数比算法更重要。** 同样的 Q-Learning 框架，换一个奖励函数——从超额-16.49% 变到 +22.15%：

| # | 奖励函数 | 策略收益 | 超额收益 | 买卖分布 |
|---|---------|---------|---------|---------|
| 1 | 方向性 ±1 | +16.03% | **-5.76%** | 171买/230卖 |
| 2 | **幅度×2(无摩擦)** | **+43.94%** | **+22.15%** | **125买/387卖** |
| 3 | 不对称买×3/卖×1 | +5.30% | **-16.49%** | 94买/449卖 |
| 4 | 阈值触发>0.5% | +14.32% | **-7.47%** | 92买/193卖 |

---

### 四个奖励函数逐一拆解

每个奖励函数是一段 Python 代码，定义了"如果明天涨/跌这么多，买入/卖出/持有分别该得多少分"。以下是 4 个函数的完整设计和意图：

---

#### ① 方向性 ±1（去漂移）

```python
买入 → +1 如果明天涨，-1 如果明天跌
卖出 → +1 如果明天跌，-1 如果明天涨
持有 → 0
```

**只关心方向**：明天涨 0.1% 和涨 5% 都拿到 +1 分，涨跌幅度全部丢弃。

买入持有收益 **+21.79%**，策略只赚 **+16.03%**，超额 **-5.76%**。因为幅度的信息全浪费了。微涨微跌也都是 ±1 分，agent 在无意义的波动上也积极交易，交易成本吃掉收益。

---

#### ② 幅度×2(无摩擦) — ✅ 最优

```python
买入 → 明天涨跌幅 × 200%   例如：明天涨1% → +2 分，涨5% → +10 分
卖出 → -明天涨跌幅 × 200%  例如：明天跌1% → +2 分，跌5% → +10 分
持有 → 0
```

**"无摩擦" = 持有得 0 分（中性基准），且交易动作不额外扣手续费。** Agent 不需要为"动手交易"本身付出代价，只需要判断方向对不对、幅度够不够大。

策略 **+43.94%** vs 买持 **+21.79%**，超额 **+22.15%**。持有=0 意味着"没把握就老实呆着"——上证指数大部分时间波动小，agent 只在涨跌超过一定幅度时才出手。卖远多于买（387 卖出 / 125 买入），完美匹配震荡市中"高抛低吸"的需要。

---

#### ③ 不对称(买激进/卖保守)

```python
买入 → 明天涨跌幅 × 800%   买入正确拿8倍分
卖出 → -明天涨跌幅 × 300%  卖出正确只拿3倍分
持有 → 0
```

**故意制造偏见**：买入做对了收益巨大，卖出做对了收益一般。Agent 天然更愿意买、不太愿意卖。

策略 **+5.30%**，超额 **-16.49%**。这个奖励是为强趋势股设计的（个股有研新材用它超额 +206%），但在指数上买偏见让 agent 在震荡市中不断追涨→被套，收益还不如不动。**同一函数在不同资产上天差地别。**

---

#### ④ 阈值触发(>0.5%)

```python
如果|明天涨跌幅| > 0.5%：  方向正确 +2，方向错误 -2
如果|明天涨跌幅| ≤ 0.5%：  方向正确 +0.5，方向错误 -0.5
持有 → 0
```

**一刀切阈值**：0.5% 以上的波动按统一强度奖罚，0.5% 以下的波动统一弱化。连续的价格变化被硬切成两档。

策略 **+14.32%**，超额 **-7.47%**。阈值丢失了 0.3-0.8% 区间的细节信息，而这是指数很多有效交易信号所在的区间。交易频率也偏低（285 次交易，其余 671 天持有），过于保守。

---

**总结：四者的核心差别是"对幅度的信息利用程度"**

| 奖励函数 | 是否利用幅度 | 持有基准 | 结果 |
|---------|------------|---------|------|
| 方向性 ±1 | ❌ 只看正负 | 0（中立） | 超额 **-5.76%** |
| 阈值触发 | ⚠️ 砍成两档 | 0（中立） | 超额 **-7.47%** |
| **幅度×2(无摩擦)** | **✅ 完整保留** | **0（中立）** | **超额 +22.15%** |
| 不对称 | ✅ 完整但有偏 | 0（中立） | 超额 **-16.49%** |

## 结果总览

| 指标 | 数据 |
|------|------|
| 训练期（2022-2026）策略收益 | **+43.94%** |
| 同期买入持有 | +21.79% |
| 超额收益 | **+22.15%** |
| OOT 测试期（2018-2022）策略收益 | **+35.42%** |
| OOT 同期买入持有 | -2.42% |
| OOT 超额收益 | **+37.84%** |
| OOT 买入命中率 | 56% |
| OOT 卖出命中率 | 54% |
| 交易频率 | 512次/956天 ≈ 54% |
| 参数总量 | **36 个浮点数** |
| 训练耗时 | ~0.3 秒（MacBook CPU）|

→ **OOT（Out-of-Time）测试**：在训练数据之前（2018-2022）的完全未见过回测中，策略依然正超额 **+37.84%**，买入/卖出准确率保持在 54-56%，证明学到的 Q 表没有过拟合训练期的市场结构。

## 技术原理

### Tabular Q-Learning（查表法）

```
Q[state, action] += α × (reward + γ × max(Q[next_state]) - Q[state, action])
```

- **状态 S**（12 个）：每日涨跌幅落入 1% 区间，从 `<-5%` 到 `>=5%`
- **动作 A**（3 个）：买入、卖出、持有
- **Q 表大小**：12 × 3 = **36 个浮点数**

这是最经典的**查表式 Q-Learning**，不是 DQN/PPO 等神经网络方法。36 个参数、0.3 秒训练完——因为状态空间是离散的且极小（1 维输入→12 个 bin），不需要神经网络去近似 Q 函数。每一格 Q[s][a] 都可以直观解释："今天涨了 X%，过去 1000 轮模拟中，买入/卖出/持有哪个长期回报最高？"

### Q 表策略画像（最优权重）

```
     状态       买入       卖出       持有     → 动作   训练期出现次数
   <-5%       +9.27     +18.19     +10.87   → 卖出         2天
[-5%,-4%)     0.00       0.00       0.00   → (未访问)       0天
[-4%,-3%)    +18.62     +10.28     +12.00   → 买入         2天
[-3%,-2%)    +17.51     +16.96     +17.08   → 买入        16天
[-2%,-1%)    +17.09     +17.14     +17.28   → 持有        74天
[-1%,0%)     +16.98     +17.15     +16.93   → 持有       379天
  [0%,1%)    +16.98     +17.12     +17.16   → 卖出       393天
  [1%,2%)    +17.42     +17.12     +17.16   → 买入        90天
  [2%,3%)    +19.75     +16.10     +18.11   → 买入        14天
  [3%,4%)    +25.19      +6.98     +12.34   → 买入         3天
  [4%,5%)    +22.50      +5.67      +9.67   → 买入         2天
   >=5%      -0.56      +33.41      +6.43   → 卖出         1天
```

解读：**绝大多时间（79% 的交易日）涨跌幅在 ±1% 以内**。在这个区间，`[-1%,0%)→持有`、`[0%,1%)→卖出`——小涨就卖，微跌不动，震荡市里赚小幅波动的钱。而 **大涨 2-5% 坚决买入**（追涨信号），**暴跌 <-5% 和 ≥5% 卖出**（避险）。整体 125买/387卖，卖远多于买，是一种偏保守的震荡市策略。

### 为什么不需要神经网络

| 对比项 | 本策略 (Tabular Q) | DQN / PPO |
|--------|------------------|-----------|
| 参数数量 | 36 | 数百万 |
| 状态空间 | 12 个离散值 | 连续/高维 |
| 训练硬件 | 任意 CPU | GPU 推荐 |
| 训练时间 | < 1 秒 | 分钟~小时 |
| 可解释性 | 每格数值一目了然 | 黑盒权重 |
| 是否过拟合风险 | 极低（参数极少） | 高（需大量正则化） |

**12 个离散状态 × 3 个动作 = 36 个参数**，直接查表即可，不需要神经网络去近似 Q 函数。MPS/GPU 启动开销都比训练总量大几个数量级。

## 快速开始

```bash
# 1. 安装依赖
pip install numpy pandas akshare matplotlib

# 2. 训练（~0.3 秒）
python q_learning_index.py

# 3. 奖励函数对比（选出最优）
python compare_rewards.py

# 4. OOT 泛化测试
python oot_test.py

# 5. 训练 vs OOT 择时质量对比
python oot_detail.py

# 6. 今日信号生成
python signal.py
```

### 依赖

```
numpy
pandas
akshare      # A 股数据接口
matplotlib   # OOT 可视化
```

### 数据说明

训练代码使用 `akshare` 库获取上证指数日线数据。训练区间 **2022-06-23 ~ 2026-07-03**，数据自动从新浪财经拉取，无需额外配置。

## 项目结构

```
q-learning-shanghai-index/
├── README.md                   ← 本文件
├── q_learning_index.py         # 主训练脚本（含回测 + 今日信号）
├── compare_rewards.py          # 4 种奖励函数对比
├── oot_test.py                 # OOT 泛化测试
├── oot_detail.py               # 训练 vs OOT 择时质量详细对比
├── oot_chart.py                # OOT 买卖信号可视化
├── signal.py                   # 每日信号生成器
├── REWARD_COMPARISON.md        # 奖励函数全对比报告
├── oot_index_actions.png       # OOT 买卖信号可视化图
├── weights/
│   ├── Q_table_index_best.npy  # 最优 Q 表权重（36 个浮点数）
│   └── metadata_index_best.json # 训练元数据
└── requirements.txt
```

## 关键发现

1. **奖励函数 > 算法结构**：同样的 Q-Learning 框架，最优奖励超额 **+22.15%**，最差的 **-16.49%**，差值达 38 个百分点
2. **无摩擦（持有=0）是关键**：对指数这种低波动资产，去除交易摩擦基准让 agent 只在方向明确时才交易
3. **OOT 泛化优秀**：2018-2022 年（-2.42% 熊市→震荡）超额 **+37.84%**，买入准确率 56%、卖出准确率 54%，证明没有过拟合
4. **36 个参数足够**：当状态空间只有 1 维（涨跌幅）× 12 个 bin 时，无需神经网络
5. **不同资产需不同奖励**：对比个股策略（不对称奖励买入×3/卖出×1），指数最优是无摩擦比例奖励——奖励函数必须匹配资产的趋势结构

## QLens Agent — AI 市场分析助手

> 基于 Google ADK (Agent Development Kit) 构建的 LLM Agent，连接到本地的 Q-Learning 模型服务，用自然语言回答市场预测相关的问题。

QLens Agent 是一个 LLM Agent，它连接你的本地 Q-Learning 模型服务（端口 8765），能够：

- 📊 **自动获取最新 dashboard**：每次对话前自动注入模型 dashboard 数据，LLM 无需额外调用工具即可感知市场状态
- 🔮 **生成新预测**：通过 `generate_prediction` 工具触发模型生成最新的买/卖/持有信号
- 📈 **评估历史预测**：通过 `evaluate_pending` 工具检查过去预测的准确率和模型奖励更新
- 🗣️ **自然语言交互**：用中文对话即可了解市场信号、Q 值、置信度等

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    QLens Agent (app/)                    │
│                                                         │
│  ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  FastAPI     │    │  ADK Agent                   │   │
│  │  (端口 8080)  │    │                              │   │
│  │  · ADK API   │───▶│  · before_model_callback    │   │
│  │  · A2A JSON- │    │    (自动注入 dashboard)      │   │
│  │    RPC       │    │  · tools:                   │   │
│  │  · /feedback │    │    - generate_prediction    │   │
│  └─────────────┘    │    - evaluate_pending        │   │
│                     └───────┬──────────────────────┘   │
│                             │                           │
│                     ┌───────▼──────────────────────┐   │
│                     │  LLM Router (端口 8046)        │   │
│                     │  · model: gpt-4o-mini         │   │
│                     └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │
         ▼  HTTP (localhost)
┌─────────────────────────────────────────────────────────┐
│              Q-Learning Model Server (端口 8765)          │
│  · /api/dashboard  — 获取最新 dashboard                  │
│  · /api/predict    — 生成新预测                           │
│  · /api/evaluate   — 评估待定预测                         │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites

QLens Agent 依赖两个本地服务：

| 服务            | 端口  | 说明                              |
|---------------|------|---------------------------------|
| LLM Router    | 8046 | OpenAI 兼容的 LLM 路由（Qwen32B 等）    |
| Market API    | 8765 | Q-Learning 模型服务（运行训练好的 Q 表权重）  |

确保这两个服务在运行后再启动 Agent。

### 快速开始

```bash
# 1. 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步依赖
cd app
uv sync

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等配置

# 4. 启动 Agent（开发模式）
uv run uvicorn app.fast_api_app:app --reload --port 8080

# 5. 访问 API
#    · Swagger UI:          http://localhost:8080/docs
#    · ADK Web UI:          http://localhost:8080/
#    · Agent Card (A2A):    http://localhost:8080/a2a/app/.well-known/agent-card
```

### 测试

```bash
# 运行 E2E 测试（InMemorySessionService，不依赖外部服务）
uv run pytest tests/ -v
```

测试使用 `InMemorySessionService`，不依赖 Market API 或 LLM Router，仅验证 Agent 定义和工具注册是否正确。

### 环境变量

| 变量                     | 默认值                          | 说明                               |
|------------------------|--------------------------------|----------------------------------|
| `OPENAI_BASE_URL`      | `http://127.0.0.1:8046/v1`    | LLM Router 地址                    |
| `OPENAI_API_KEY`       | —                              | API Key（必填）                      |
| `MARKET_API_BASE`      | `http://127.0.0.1:8765`       | Q-Learning 模型服务地址               |
| `GOOGLE_CLOUD_PROJECT` | —                              | GCP 项目 ID（部署时必填）               |
| `GOOGLE_CLOUD_LOCATION`| `global`                       | GCP 区域                            |
| `LOGS_BUCKET_NAME`     | —                              | GCS bucket（用于 prompt-response 日志） |
| `APP_URL`              | `http://0.0.0.0:8000`         | A2A Agent Card 公布的 URL            |

### API 端点

| 端点                                      | 方法   | 说明                    |
|------------------------------------------|------|-----------------------|
| `/`                                      | GET  | ADK Web UI             |
| `/api/`                                  | GET  | ADK API 根路径           |
| `/a2a/app/.well-known/agent-card`        | GET  | A2A Agent Card（用于注册）  |
| `/a2a/app`                               | POST | A2A JSON-RPC 端点       |
| `/feedback`                              | POST | 收集用户反馈               |
| `/docs`                                  | GET  | Swagger UI 文档         |

### Agent 工具

| 工具                    | 说明                              |
|-----------------------|---------------------------------|
| `get_dashboard`       | 获取最新 dashboard（含信号、Q 值、模型性能）    |
| `generate_prediction` | 触发 Q-Learning 模型生成新预测            |
| `evaluate_pending`    | 评估过去的预测是否准确，更新模型奖励             |

注：每次 LLM 调用前，`before_model_callback` 会自动获取并注入 dashboard 数据到系统提示中，因此无需显式调用 `get_dashboard`。

### 部署选项

#### 方式一：Agent Registry（推荐）

```bash
# 构建 Docker 镜像并推送
docker build -t gcr.io/your-project/qlens-agent:latest .
docker push gcr.io/your-project/qlens-agent:latest

# 部署到 Cloud Run
gcloud run deploy qlens-agent \
  --image gcr.io/your-project/qlens-agent:latest \
  --port 8080 \
  --execution-environment gen2 \
  --set-env-vars "OPENAI_BASE_URL=...,OPENAI_API_KEY=...,MARKET_API_BASE=..."

# 发布到 Agent Registry
agents-cli publish gemini-enterprise
```

部署后 Agent 可通过 A2A 协议被 Gemini Enterprise 或其他 A2A 客户端调用。

#### 方式二：Cloudflare Tunnel（公开访问）

如果想把本地开发的 Agent 临时暴露到公网（例如给朋友测试），可以使用 Cloudflare Tunnel：

```bash
# 安装 cloudflared
brew install cloudflared

# 创建 Tunnel（指向本地 8080）
cloudflared tunnel --url http://localhost:8080
```

这会生成一个 `https://xxx.trycloudflare.com` 的公开 URL。把此 URL 设为 `APP_URL`，即可让 Agent Card 指向公网地址。

### 项目结构

```
app/                          ← QLens Agent 代码
├── agent.py                  # Agent 定义 + before_model_callback
├── tools.py                  # 工具函数（dashboard, predict, evaluate）
├── fast_api_app.py           # FastAPI 应用入口 + A2A 路由
├── app_utils/
│   ├── a2a.py                # A2A 端点注册
│   ├── services.py           # Session/Artifact 服务
│   ├── telemetry.py          # OpenTelemetry 遥测
│   └── typing.py             # Feedback 模型
├── __init__.py               # 导出 app
tests/
├── test_e2e.py               # E2E 测试（InMemorySessionService）
└── __init__.py
Dockerfile                    # 容器化部署（uv + uvicorn）
pyproject.toml                # 依赖和项目元数据
uv.lock                       # 锁定依赖版本
.env.example                  # 环境变量模板
```

## 相关项目

- 个股版（有研新材 600206）：同期超额 **+205.72%**，使用不对称奖励（买入×3/卖出×1）
