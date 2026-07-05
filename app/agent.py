# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QLens Agent — A Q-Learning market prediction analyst.

Connects to a local Q-Learning model server (http://127.0.0.1:8765) that
tracks the Shanghai Composite Index and provides buy/sell/hold signals
with confidence scores.

Architecture:
  The agent uses ``before_model_callback`` to automatically inject the
  latest dashboard data into every LLM call, plus registered tools
  (generate_prediction, evaluate_pending) for on-demand actions. The LLM
  runs via the local OpenAI-compatible router at port 8046 with
  gpt-4o-mini → OpenAILlm.

Environment Variables:
  OPENAI_BASE_URL   —  OpenAI-compatible router endpoint (default: http://127.0.0.1:8046/v1)
  OPENAI_API_KEY    —  API key for the router (required; set in .env)
  MARKET_API_BASE   —  Q-Learning model server URL (default: http://127.0.0.1:8765)
"""

import os

# Point OpenAILlm at the local router (OpenAI-compatible proxy).
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8046/v1")

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

from app.tools import get_dashboard, generate_prediction, evaluate_pending


async def inject_market_data(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Fetch market dashboard and inject it into the LLM context.

    Called before every LLM invocation. Fetches the Q-Learning model's
    dashboard and prepends it as a system message so the LLM has the
    latest market data available without needing function calling.
    """
    try:
        dashboard_json = await get_dashboard()
    except Exception as e:
        dashboard_json = f'{{"error": "Failed to fetch market data: {e}"}}'

    # Inject the market dashboard as a system-level context message
    context_part = Part(
        text=(
            "Here is the latest data from the Q-Learning market model:\n\n"
            f"{dashboard_json}\n\n"
            "Use this data to answer the user's question. Explain the "
            "prediction signal, confidence, and metrics in plain Chinese."
        )
    )
    context_content = Content(role="system", parts=[context_part])
    llm_request.contents.insert(0, context_content)

    return None  # Continue to LLM with modified request


root_agent = Agent(
    name="root_agent",
    model="gpt-4o-mini",  # via local router → OpenAILlm
    instruction=(
        "You are a Q-Learning market analyst for the Shanghai Composite Index. "
        "Your task is to explain the latest prediction signal (buy/sell/hold), "
        "confidence margin, and Q-values in plain Chinese. "
        "If the dashboard data includes an error, tell the user the market "
        "model server is unavailable.\n\n"
        "The model uses Q-Learning with a state space based on technical "
        "indicators. A high confidence margin (>0.3) means the model is "
        "more certain about its signal. The action space is: buy (做多), "
        "sell (做空), hold (持有).\n\n"
        "You have tools available: generate_prediction (triggers a fresh "
        "prediction from the Q-Learning model) and evaluate_pending "
        "(evaluates past predictions against actual market moves). "
        "Use them when the user asks for a new prediction or wants to "
        "check how the model is performing."
    ),
    before_model_callback=inject_market_data,
    tools=[generate_prediction, evaluate_pending],
)


from google.adk.apps import App

app = App(root_agent=root_agent, name="app")
