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
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import wraps
from time import perf_counter, time
from typing import TYPE_CHECKING, Any, List, Optional

from fastapi import Body, FastAPI, Request
from pydantic import ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import (
    AgentServerRef,
    ModelServerRef,
    ResourcesServerRef,
    ROLLOUT_PATH_PREFIX,
)
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body, rollout_context
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    get_response_json,
    raise_for_status,
    rollout_path_prefix,
)

if TYPE_CHECKING:
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


class BaseEpisodeProcessorConfig(BaseRunServerInstanceConfig):
    agent: Optional[AgentServerRef] = None
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    max_steps: Optional[int] = None


class BaseEpisodeProcessorServer(BaseServer):
    config: BaseEpisodeProcessorConfig

    @abstractmethod
    async def run(self, request: Optional[Request] = None, body: BaseRunRequest = Body()) -> NeMoGymResponse:
        pass



class RLVREpisodeProcessorServer(BaseEpisodeProcessorServer, AggregateMetricsMixin, SimpleServer):
    config: BaseEpisodeProcessorConfig


    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)

        run = self.run

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> BaseVerifyResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, BaseRunRequest)), None)
            with rollout_context(self.rollout_id_from_run(body)):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def _model_call_capture_enabled(self) -> bool:
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        if not self._model_call_capture_enabled():
            return None
        return maybe_rollout_id_from_run_body(body)

    def url_path_for_run(self, url_path: str, body: Any) -> str:
        return f"{rollout_path_prefix(self.rollout_id_from_run(body))}{url_path}"

    async def run(self, request: Optional[Request] = None, body: BaseRunRequest = Body()) -> NeMoGymResponse:
        server_client = self.server_client
        config = self.config

        agent_ref = getattr(config, "agent", None)
        agent_name = agent_ref.name if hasattr(agent_ref, "name") else (agent_ref or config.name)

        resources_server_ref = getattr(config, "resources_server", None)
        resources_server_name = (
            resources_server_ref.name if hasattr(resources_server_ref, "name") else resources_server_ref
        )

        cookies = request.cookies if request is not None else {}

        # 1. Seed session on Resources Server if defined
        if resources_server_name:
            seed_session_response = await server_client.post(
                server_name=resources_server_name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies

        # 2. Run interaction loop between Agent and Resources Server
        expected_rollout_id = self.rollout_id_from_run(body)
        collect_trajectory = self._model_call_capture_enabled() and expected_rollout_id is not None
        rollout_id = expected_rollout_id or "unscoped"

        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._run_loop(
            request=request,
            body=body,
            agent_name=agent_name,
            resources_server_name=resources_server_name,
            cookies=cookies,
            rollout_id=rollout_id,
            collect_trajectory=collect_trajectory,
        )

        cookies = dict(*cookies.items(), *resources_server_cookies.items(), *model_server_cookies.items())

        if trajectory is not None:
            extra = getattr(body, "model_extra", {}) or {}
            task_id = next(
                (
                    str(extra[key])
                    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                    if extra.get(key) is not None
                ),
                "unknown",
            )
            trajectory = trajectory.model_copy(
                update={
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "turns": [
                        turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id})
                        for turn in trajectory.turns
                    ],
                }
            )

        # 3. Verify task on Resources Server if defined
        if resources_server_name:
            verify_payload = body.model_dump() | {"response": model_response.model_dump(mode="json")}
            verify_response = await server_client.post(
                server_name=resources_server_name,
                url_path="/verify",
                json=verify_payload,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            result = await get_response_json(verify_response)

            reward = result.get("reward", 0.0)
            if trajectory is not None:
                resolved = result.get("resolved")
                if isinstance(resolved, bool) and trajectory.turns:
                    trajectory.turns[-1].resolved = resolved
                else:
                    trajectory.gaps.append(ObservationGap(code="resolution_unavailable", invocation_id="root"))
                result["ng_trajectory"] = trajectory.model_dump(mode="json")

            extra_updates = {
                "reward": reward,
                **{k: v for k, v in result.items() if k not in ("responses_create_params", "response")},
            }
            return model_response.model_copy(update=extra_updates)

        return model_response


    async def _run_loop(
        self,
        request: Optional[Request],
        body: BaseRunRequest,
        agent_name: str,
        resources_server_name: Optional[str],
        cookies: Any,
        rollout_id: str,
        collect_trajectory: bool,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        server_client = self.server_client
        config = self.config
        max_steps = getattr(config, "max_steps", None)

        invocation_id = "root"
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        trajectory_gaps: list[ObservationGap] = []

        params = body.responses_create_params.model_copy(deep=True)
        if isinstance(params.input, str):
            params.input = [NeMoGymEasyInputMessage(role="user", content=params.input)]

        new_outputs = []
        usage = None
        step = 0
        invocation_status = "completed"
        model_server_cookies = cookies
        resources_server_cookies = cookies

        url_path_for_responses = self.url_path_for_run("/v1/responses", body)

        while True:
            step += 1
            new_params = params.model_copy(update={"input": params.input + new_outputs})
            if collect_trajectory:
                turn_timestamp = time()

            # Call Agent server via POST /v1/responses
            agent_response = await server_client.post(
                server_name=agent_name,
                url_path=url_path_for_responses,
                json=new_params,
                cookies=model_server_cookies,
            )
            await raise_for_status(agent_response)
            agent_response_json = await get_response_json(agent_response)
            model_server_cookies = agent_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(agent_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from agent server: {json.dumps(agent_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)
            if collect_trajectory:
                turn_model_calls = []
                if model_response.id:
                    model_server_ref = getattr(config, "model_server", None)
                    model_call_ref = ModelCallRef(model_ref=model_server_ref, response_id=model_response.id)
                    model_calls.append(model_call_ref)
                    turn_model_calls.append(model_call_ref)
                else:
                    trajectory_gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable", invocation_id=invocation_id, detail=f"turn:{step}"
                        )
                    )
                reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
                answer = [item for item in output if item.type != "reasoning"]
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation_id,
                        task_id="unscoped",
                        rollout_id=rollout_id,
                        turn_no=step,
                        timestamp=turn_timestamp,
                        question=new_params.input,
                        answer=answer,
                        reasoning_content=reasoning,
                        step_count=len(tool_records),
                        model_calls=turn_model_calls,
                    )
                )

            usage = accumulate_response_usage(usage, model_response.usage)
            model_response.usage = None

            if model_response.incomplete_details:
                invocation_status = "incomplete"
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                    if collect_trajectory:
                        error_type = type(e).__name__
                        tool_status = "failed"
                else:
                    if resources_server_name:
                        api_response = await server_client.post(
                            server_name=resources_server_name,
                            url_path=f"/{output_function_call.name}",
                            json=parsed_arguments,
                            cookies=resources_server_cookies,
                        )
                        tool_output = (await api_response.content.read()).decode()
                        resources_server_cookies = api_response.cookies
                        if collect_trajectory:
                            completed = 200 <= api_response.status < 400
                            tool_status = "completed" if completed else "failed"
                            error_type = None if completed else f"http_{api_response.status}"
                    else:
                        tool_output = json.dumps({"error": "No resources_server configured to execute tools"})
                        tool_status = "failed"
                        error_type = "no_resources_server"

                if collect_trajectory:
                    tool_records.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=output_function_call.call_id,
                            tool_name=output_function_call.name,
                            started_at=started_at,
                            completed_at=max(started_at, time()),
                            duration_ms=(perf_counter() - started_monotonic) * 1000,
                            timing_source="executor",
                            status=tool_status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )

                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=tool_output,
                    )
                )

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

            if max_steps and step >= max_steps:
                invocation_status = "incomplete"
                break

        model_response.output = new_outputs
        model_response.usage = usage
        trajectory = None
        if collect_trajectory:
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=[*params.input, *new_outputs],
            )
            trajectory = TrajectoryRecord(
                task_id="unscoped",
                rollout_id=rollout_id,
                invocations=[invocation],
                turns=turns,
                tool_calls=tool_records,
                gaps=trajectory_gaps,
            )
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        resources_server_ref = getattr(self.config, "resources_server", None)
        resources_server_name = (
            resources_server_ref.name if hasattr(resources_server_ref, "name") else resources_server_ref
        )
        if resources_server_name:
            response = await self.server_client.post(
                server_name=resources_server_name,
                url_path="/aggregate_metrics",
                json=body,
            )
            await raise_for_status(response)
            return AggregateMetrics.model_validate(await get_response_json(response))

        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )


SimpleEpisodeProcessorServer = RLVREpisodeProcessorServer



class SimpleEpisodeProcessorAdapter:
    """Helper adapter to allow Agent servers to delegate run() to an episode processor."""

    def __init__(self, agent: "SimpleResponsesAPIAgent"):
        self.agent = agent

    async def run(self, request: Optional[Request], body: BaseRunRequest) -> NeMoGymResponse:
        processor_server = SimpleEpisodeProcessorServer(
            config=BaseEpisodeProcessorConfig(
                host=self.agent.config.host,
                port=self.agent.config.port,
                entrypoint=self.agent.config.entrypoint,
                name=self.agent.config.name,
                agent=AgentServerRef(type="responses_api_agents", name=self.agent.config.name),
                resources_server=getattr(self.agent.config, "resources_server", None),
                model_server=getattr(self.agent.config, "model_server", None),
                max_steps=getattr(self.agent.config, "max_steps", None),
            ),
            server_client=self.agent.server_client,
        )
        return await processor_server.run(request, body)
