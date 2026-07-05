"""Tools for querying the Q-Learning market prediction API.

Each function is automatically registered as an ADK tool via the Agent's
tools parameter. Args and docstrings define the tool schema for the LLM.
"""

import json
import os

import httpx

API_BASE = os.getenv("MARKET_API_BASE", "http://127.0.0.1:8765")


async def get_dashboard(asset: str = "index") -> str:
    """Fetch the latest market dashboard data from the Q-Learning model.

    Returns current market status, the latest prediction signal (buy/sell/hold),
    Q-values, confidence metrics, and model performance statistics.

    Args:
        asset: Asset identifier. Defaults to "index" for Shanghai Composite.

    Returns:
        JSON string with dashboard fields: summary, latest signal, records,
        model info, and ledger.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{API_BASE}/api/dashboard", params={"asset": asset}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    return json.dumps(data, indent=2, ensure_ascii=False)


async def generate_prediction(asset: str = "index") -> str:
    """Trigger a new prediction from the Q-Learning model.

    The model analyzes recent market data and outputs an action signal
    (buy / sell / hold) along with Q-value estimates and confidence margin.

    Args:
        asset: Asset identifier. Defaults to "index" for Shanghai Composite.

    Returns:
        JSON string with the prediction result including action, confidence,
        and analysis timestamp.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_BASE}/api/predict", params={"asset": asset}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    return json.dumps(data, indent=2, ensure_ascii=False)


async def evaluate_pending(asset: str = "index") -> str:
    """Evaluate pending Q-Learning predictions against actual market moves.

    Checks if previous predictions were correct, updates the model's
    learning ledger, and returns performance metrics.

    Args:
        asset: Asset identifier. Defaults to "index" for Shanghai Composite.

    Returns:
        JSON string with evaluation results showing prediction accuracy
        and model reward updates.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_BASE}/api/evaluate", params={"asset": asset}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    return json.dumps(data, indent=2, ensure_ascii=False)
