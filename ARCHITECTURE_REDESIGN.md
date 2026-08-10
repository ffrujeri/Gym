# NeMo-Gym Architecture Redesign Specification & Walkthrough

## 1. Executive Summary & Motivation

In previous iterations of the NeMo-Gym framework, `BaseResponsesAPIAgent` conflated two distinct responsibilities:
1. **Agent Policy Generation (`POST /v1/responses`)**: Transforming prompt inputs into candidate outputs (or running internal sub-agent / CLI loops).
2. **Episode Rollout Orchestration (`POST /run`)**: Seeding environment state (`/seed_session` or `/reset`), driving multi-turn tool steps, parsing observations, and scoring outcomes (`/verify` or `/step`).

This conflation forced every agent (e.g. `simple_agent`, `claude_code_agent`, `gymnasium_agent`) to duplicate environment orchestration code, mix tool-stepping loops with policy calls, and produce inconsistent verify response schemas.

The new architecture introduces a top-level **`episode_processors/`** layer that decouples **Episode Rollout Orchestration** from **Agent Policy Logic**.

---

## 2. Core Concepts

### Turn Concepts in Agentic Tasks
- **User Turn (Task Specification)**: The task specification prompt provided to the agent via `NeMoGymResponseCreateParamsNonStreaming.input`.
- **Assistant Turn (Episode Collection)**: The agent's execution of the task against model backends or environment tools.

### Architectural Principles

```
                                  [ Gym CLI / rollout_collection ]
                                                  |
                                             POST /run
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                        episode_processors/                                        |
|                          (Dedicated Rollout Orchestrator hosting POST /run)                       |
|                                                                                                   |
|  POST /run                                                                                        |
|    1. ResourcesServer.post("/seed_session", body) --------> Seeds environment session             |
|    2. Agent.post("/v1/responses", body.responses_create_params) -> Executes Assistant turn        |
|         - Macro / Self-contained Agent: executes episode & returns NeMoGymResponse                |
|         - Micro Agent: returns tool call choices; EpisodeProcessor steps tools until done         |
|    3. Assembles Trajectory & Observability records                                                |
|    4. ResourcesServer.post("/verify", response) ----------> Evaluates task reward & metrics       |
+---------------------------------------------------------------------------------------------------+
                               |                                                 |
                               v                                                 v
+----------------------------------------------+  +-------------------------------------------------+
|             responses_api_agents/            |  |               resources_servers/                |
|          (Pure Policy / Agent Harness)       |  |              (Task Environment)                  |
|                                              |  |                                                 |
|  POST /v1/responses                          |  |  POST /seed_session                             |
|    - Accepts User Turn (task specification)  |  |  POST /{tool_name}                              |
|    - Runs Assistant Turn (model / CLI loop)  |  |  POST /verify                                   |
|    - Returns completed NeMoGymResponse       |  +-------------------------------------------------+
+----------------------------------------------+
```

---

## 3. Directory Layout & Component Roles

```
/home/ffruj/Gym/
├── episode_processors/                       # [NEW] Rollout Orchestration Servers (POST /run)
│   ├── rlvr_episode_processor/
│   │   └── app.py                            # Verifiable Outcome Reward Processor (/seed_session -> /verify)
│   ├── gymnasium_episode_processor/          # Traditional Step-by-Step Gym Processor (/reset -> /step loop)
│   └── simple_episode_processor/             # Alias for rlvr_episode_processor
├── responses_api_agents/                     # Pure Policy Agents (POST /v1/responses)
│   ├── simple_agent/                         # Shared Unified Model Proxy Agent
│   └── claude_code_agent/                    # Claude Code CLI Agent Harness
├── resources_servers/                        # Task Environments
└── responses_api_models/                     # Model Inference Backends
```

---

## 4. The Two Episode Processor Strategies

| Dimension | **`rlvr_episode_processor`** | **`gymnasium_episode_processor`** |
| :--- | :--- | :--- |
| **Target Paradigm** | RLVR / Verifiable Task Environments | Traditional Step-by-Step Gymnasium Environments |
| **Interaction Contract** | `/seed_session` $\rightarrow$ tool calls $\rightarrow$ `/verify` | `/reset` $\rightarrow$ `/step` transition loop |
| **Reward Signal** | **Sparse Outcome Reward**: Evaluated at episode end via `POST /verify` | **Dense Step Reward**: Accumulated step-by-step until `terminated` or `truncated` |
| **Target Tasks** | Math, Code, SWE-bench, WebArena, QA | CartPole, Atari, MuJoCo, ALFWorld, TextWorld |

---

## 5. Unified Data Output Contract (`NeMoGymResponse`)

All Episode Processors return a single, unified output format: **`NeMoGymResponse`**.

### `NeMoGymResponse` Schema
- **`output`**: `List[NeMoGymResponseOutputItem]` — Complete sequence of trajectory turns (User prompt, Assistant messages, function calls, function call outputs, reasoning).
- **`usage`**: `NeMoGymResponseUsage` — Cumulative input, output, and reasoning token usage across all turns.
- **`reward`**: `float` — Final reward score.
- **`ng_trajectory`**: `dict` — Trajectory record (turn timings, tool durations, gaps).
- **`resolved`**: `Optional[bool]` — Task completion flag.
- **`terminated` / `truncated` / `info`**: Gymnasium transition flags (when applicable).

---

## 6. YAML Configuration Examples

```yaml
episode_processors:
  # RLVR Math Evaluation Benchmark
  math_eval_runner:
    entrypoint: episode_processors/rlvr_episode_processor/app.py
    agent:
      type: responses_api_agents
      name: simple_agent
    resources_server:
      type: resources_servers
      name: math_eval_resources_server

  # Traditional Gymnasium Benchmark
  cartpole_runner:
    entrypoint: episode_processors/gymnasium_episode_processor/app.py
    agent:
      type: responses_api_agents
      name: simple_agent
    resources_server:
      type: resources_servers
      name: cartpole_resources_server

responses_api_agents:
  simple_agent:
    entrypoint: responses_api_agents/simple_agent/app.py
    model_server:
      type: responses_api_models
      name: policy_model

resources_servers:
  math_eval_resources_server:
    entrypoint: resources_servers/math_eval/app.py
  cartpole_resources_server:
    entrypoint: resources_servers/cartpole/app.py
```

---

## 7. Migration Guide

1. **Legacy Agent Servers**:
   - Legacy Agent servers delegating `run()` automatically use `SimpleEpisodeProcessorAdapter`, which proxies `run()` to `RLVREpisodeProcessorServer` for complete backward compatibility.
2. **New Agents**:
   - Only implement `POST /v1/responses` by subclassing `SimpleResponsesAPIAgent`.
3. **New Benchmarks**:
   - Define a task environment under `resources_servers/`.
   - Wire it in YAML using `episode_processors/rlvr_episode_processor/app.py` or `episode_processors/gymnasium_episode_processor/app.py`.
