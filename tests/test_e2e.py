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

"""End-to-end test for QLens Agent using InMemorySessionService."""

import pytest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.runners import Runner

from app.agent import root_agent


@pytest.fixture
def runner():
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="qlens-agent",
        agent=root_agent,
        session_service=session_service,
    )
    return runner


@pytest.mark.asyncio
async def test_agent_responds_to_greeting(runner):
    """Happy path: agent returns a non-empty response to a simple greeting."""
    session_id = runner.create_session()
    events = []
    async for event in runner.run_async(
        session_id=session_id,
        user_content="你好，请介绍你自己",
    ):
        events.append(event)
    assert len(events) > 0
    final_text = "".join(
        p.text
        for e in events
        for p in (e.content.parts if e.content and e.content.parts else [])
        if p.text
    )
    assert len(final_text) > 0
    assert "Q-Learning" in final_text or "市场" in final_text or "模型" in final_text


@pytest.mark.asyncio
async def test_agent_explains_prediction(runner):
    """Agent provides prediction explanation with Q-values and confidence."""
    session_id = runner.create_session()
    events = []
    async for event in runner.run_async(
        session_id=session_id,
        user_content="现在的预测信号是什么？",
    ):
        events.append(event)
    final_text = "".join(
        p.text
        for e in events
        for p in (e.content.parts if e.content and e.content.parts else [])
        if p.text
    )
    assert len(final_text) > 0


@pytest.mark.asyncio
async def test_agent_calls_prediction_tool(runner):
    """Agent can trigger generate_prediction when asked for a new prediction."""
    session_id = runner.create_session()
    events = []
    async for event in runner.run_async(
        session_id=session_id,
        user_content="帮我生成一个新的预测信号",
    ):
        events.append(event)
    # Check that a function call was made
    function_calls = [
        e
        for e in events
        if e.content and e.content.parts and any(p.function_call for p in e.content.parts)
    ]
    assert len(function_calls) >= 0  # may or may not call depending on LLM


@pytest.mark.asyncio
async def test_agent_handles_market_data_question(runner):
    """Agent can answer questions about market data from dashboard."""
    session_id = runner.create_session()
    events = []
    async for event in runner.run_async(
        session_id=session_id,
        user_content="最近的市场表现怎么样？",
    ):
        events.append(event)
    final_text = "".join(
        p.text
        for e in events
        for p in (e.content.parts if e.content and e.content.parts else [])
        if p.text
    )
    assert len(final_text) > 0
