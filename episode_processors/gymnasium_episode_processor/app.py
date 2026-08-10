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

from typing import Optional

from fastapi import Body, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_processor import BaseEpisodeProcessorConfig, BaseEpisodeProcessorServer
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    accumulate_response_usage,
)
from nemo_gym.reward_profile import AggregateMetricsMixin
from nemo_gym.server_utils import SimpleServer, get_response_json, raise_for_status
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse


class GymnasiumEpisodeProcessorConfig(BaseEpisodeProcessorConfig):
    resources_server: ResourcesServerRef
    agent: Optional[AgentServerRef] = None
    model_server: Optional[ModelServerRef] = None
    max_steps: int = Field(10, ge=1)


class GymnasiumEpisodeProcessorServer(BaseEpisodeProcessorServer, AggregateMetricsMixin, SimpleServer):
    config: GymnasiumEpisodeProcessorConfig

    async def run(self, request: Optional[Request] = None, body: BaseRunRequest = Body()) -> NeMoGymResponse:
        server_client = self.server_client
        config = self.config
        agent_name = config.agent.name if config.agent else config.name
        resources_server_name = config.resources_server.name
        env_cookies = request.cookies if request is not None else {}
        model_url_path = self.url_path_for_run("/v1/responses", body)

        # 1. Reset environment on Resources Server
        reset_resp = await server_client.post(
            server_name=resources_server_name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=env_cookies,
        )
        await raise_for_status(reset_resp)
        reset_data = EnvResetResponse.model_validate(await get_response_json(reset_resp))
        env_cookies = reset_resp.cookies

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]
        if reset_data.observation:
            base_body.input = list(base_body.input) + [
                NeMoGymEasyInputMessage(role="user", content=reset_data.observation)
            ]

        new_outputs = []
        total_reward = 0.0
        usage = None
        model_server_cookies = None
        step_data = EnvStepResponse(terminated=False, truncated=True, reward=0.0)
        last_model_response = None
        finished = False
        max_steps = getattr(config, "max_steps", 10) or 10

        for _ in range(max_steps):
            new_body = base_body.model_copy(update={"input": base_body.input + new_outputs})

            agent_resp = await server_client.post(
                server_name=agent_name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(agent_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(agent_resp))
            model_server_cookies = agent_resp.cookies
            last_model_response = model_response

            new_outputs.extend(model_response.output)
            usage = accumulate_response_usage(usage, model_response.usage)

            step_resp = await server_client.post(
                server_name=resources_server_name,
                url_path="/step",
                json=body.model_dump() | {"response": model_response.model_dump()},
                cookies=env_cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            total_reward += step_data.reward
            env_cookies = step_resp.cookies

            if step_data.terminated or step_data.truncated:
                finished = True
                break

            for tool_output in (step_data.info or {}).get("tool_outputs", []):
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_output["call_id"],
                        output=tool_output["output"],
                    )
                )

            if step_data.observation:
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=step_data.observation))

        if not finished:
            step_data = step_data.model_copy(update={"truncated": True})

        if last_model_response:
            last_model_response.output = new_outputs
            last_model_response.usage = usage
            return last_model_response.model_copy(
                update={
                    "reward": total_reward,
                    "terminated": step_data.terminated,
                    "truncated": step_data.truncated,
                    "info": step_data.info,
                }
            )

        return NeMoGymResponse(
            id="resp_gym",
            created_at=0.0,
            model="gymnasium",
            object="response",
            output=new_outputs,
            usage=usage,
            reward=total_reward,
            terminated=step_data.terminated,
            truncated=step_data.truncated,
            info=step_data.info,
        )


if __name__ == "__main__":
    GymnasiumEpisodeProcessorServer.run_webserver()
