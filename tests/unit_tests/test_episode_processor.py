# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

import pytest
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_processor import (
    BaseEpisodeProcessorConfig,
    BaseEpisodeProcessorServer,
    SimpleEpisodeProcessorAdapter,
    SimpleEpisodeProcessorServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class MockAgent(SimpleResponsesAPIAgent):
    async def responses(self, body=...):
        return NeMoGymResponse.model_validate(
            {
                "id": "resp-1",
                "created_at": 0.0,
                "model": "test-model",
                "object": "response",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
            }
        )


class TestEpisodeProcessor:
    def test_simple_episode_processor_server_initialization(self) -> None:
        config = BaseEpisodeProcessorConfig(
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="test_processor",
            agent=AgentServerRef(type="responses_api_agents", name="mock_agent"),
            resources_server=ResourcesServerRef(type="resources_servers", name="mock_resources"),
        )
        client = MagicMock(spec=ServerClient)
        processor_server = SimpleEpisodeProcessorServer(config=config, server_client=client)
        assert processor_server.config.name == "test_processor"
        assert processor_server.config.agent.name == "mock_agent"
        assert processor_server.config.resources_server.name == "mock_resources"

    def test_simple_responses_api_agent_uses_episode_processor_adapter(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")
        agent = MockAgent(config=config, server_client=MagicMock(spec=ServerClient))
        adapter = agent.get_episode_processor()
        assert isinstance(adapter, SimpleEpisodeProcessorAdapter)
        assert adapter.agent == agent
