"""suite_runner.py — Orchestrate a full calibration suite.

Workflow:
  1. Pre-flight: resource check (refuses if insufficient memory).
  2. For each (model × profile × params combo):
       - Create a RunSpec
       - Run via LlamaRunner (start server → warmup → 3 samples → stop)
       - Aggregate samples → median
       - Persist immediately (incremental, in case of crash/abort)
       - Push progress event to subscriber callbacks
  3. After all runs: compute winners per (model, profile), persist, push complete.

Concurrency:
  - One suite runs at a time. `is_running` flag prevents overlap.
  - Abort: callers set abort_event; the loop checks between runs and inside
    LlamaRunner.

Notifications:
  - The runner takes a `notify` async callback that receives dicts:
       {"type": "suite_started",  "data": {...}}
       {"type": "suite_progress", "data": {...}}
       {"type": "run_complete",   "data": {...}}
       {"type": "suite_complete", "data": {...}}
       {"type": "suite_aborted",  "data": {...}}
       {"type": "log",            "data": {"line": str}}
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import prompts
from llama_runner import LlamaRunner, RunOutcome
from metrics import Aggregate, Sample, aggregate_parallel, median_of
from resources import check_suite_feasibility
from results_store import ResultsStore
from sweep_strategy import RunSpec, build_auto_sweep, build_manual_matrix, estimate_suite_duration_sec

log = logging.getLogger(__name__)

NotifyFn = Callable[[dict], Awaitable[None]]


class SuiteRunner:
    def __init__(self,
                 llama_bin:       Path,
                 bench_port:      int,
                 store:           ResultsStore,
                 notify:          NotifyFn,
                 n_predict:       int = 256,
                 runs_per_combo:  int = 3) -> None:
        self._bin            = llama_bin
        self._port           = bench_port
        self._store          = store
        self._notify         = notify
        self._n_predict      = n_predict
        self._runs_per_combo = runs_per_combo

        self.is_running:  bool = False
        self.current_id:  Optional[str] = None
        self._abort:      Optional[asyncio.Event] = None
        self._task:       Optional[asyncio.Task]  = None

    # -------------------------------------------------------------------
    # Public lifecycle
    # -------------------------------------------------------------------

    async def start(self,
                    selected_models: list[tuple[str, float]],   # (path, size_gb)
                    profiles:        list[str],
                    budget:          str = "standard",
                    mode:            str = "auto",
                    manual_matrix:   Optional[dict] = None,
                    draft_map:       Optional[dict] = None) -> dict:
        """Start a suite. Returns {ok, suite_id, ...} immediately.

        draft_map (auto mode): {<model_path>: <draft_path or None>, ...}
          Maps each selected model to an optional speculative-decoding draft
          GGUF. Applied only to single-profile runs. Models absent from the
          map (or mapped to None) run without a draft.

        manual_matrix (when mode="manual"):
          {
            "<model_path>": {
              "profile":     "single" | "throughput",
              "base_params": {...},
              "sweeps":      {"parallel": [...], ...},
              "prompt":      "medium",
            }
          }
        """
        if self.is_running:
            return {"ok": False, "error": "already running", "suite_id": self.current_id}

        draft_map = draft_map or {}

        # Feasibility check
        feasibility = check_suite_feasibility(
            selected_models, ctx_size=32768, parallel=24,
        )
        if not feasibility["ok"]:
            return {
                "ok":          False,
                "error":       "resources_insufficient",
                "feasibility": feasibility,
            }

        # Build all RunSpecs
        all_specs: list[RunSpec] = []
        if mode == "manual" and manual_matrix:
            for path, m in manual_matrix.items():
                all_specs.extend(build_manual_matrix(
                    model_path  = path,
                    profile     = m.get("profile", "throughput"),
                    base_params = m.get("base_params", {}),
                    sweeps      = m.get("sweeps", {}),
                    prompt_name = m.get("prompt", "medium"),
                ))
        else:
            for path, _size in selected_models:
                all_specs.extend(build_auto_sweep(
                    path, profiles, budget,
                    draft_model = draft_map.get(path),
                ))

        if not all_specs:
            return {"ok": False, "error": "no runs to execute"}

        # Create suite entry
        suite_id = ResultsStore.new_suite_id()
        self._store.start_suite(
            suite_id = suite_id,
            models   = [p for p, _ in selected_models],
            profiles = profiles,
            budget   = budget,
            mode     = mode,
        )

        self.is_running = True
        self.current_id = suite_id
        self._abort     = asyncio.Event()
        eta             = estimate_suite_duration_sec(all_specs, self._runs_per_combo)

        # Launch background task
        self._task = asyncio.create_task(self._run_loop(suite_id, all_specs, eta, feasibility))

        return {
            "ok":          True,
            "suite_id":    suite_id,
            "total_runs":  len(all_specs),
            "eta_seconds": eta,
            "feasibility": feasibility,
        }

    async def abort(self) -> bool:
        if not self.is_running or self._abort is None:
            return False
        self._abort.set()
        return True

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    async def _run_loop(self,
                        suite_id:    str,
                        specs:       list[RunSpec],
                        eta:         int,
                        feasibility: dict) -> None:
        try:
            await self._notify({
                "type": "suite_started",
                "data": {
                    "suite_id":    suite_id,
                    "total_runs":  len(specs),
                    "eta_seconds": eta,
                    "feasibility": feasibility,
                },
            })

            t_start = time.perf_counter()

            for idx, spec in enumerate(specs):
                if self._abort and self._abort.is_set():
                    # Mark not-running BEFORE notifying so the header update
                    # built inside the notify callback (which reads
                    # self.is_running) reflects the post-abort state.
                    self.is_running = False
                    await self._notify({
                        "type": "suite_aborted",
                        "data": {"suite_id": suite_id, "completed_runs": idx},
                    })
                    self._store.finish_suite(suite_id, winners={}, status="aborted")
                    return

                await self._notify({
                    "type": "suite_progress",
                    "data": {
                        "suite_id":    suite_id,
                        "run_index":   idx,
                        "total_runs":  len(specs),
                        "current":     spec.as_dict(),
                        "elapsed_sec": int(time.perf_counter() - t_start),
                    },
                })

                run_data = await self._execute_run(spec)
                self._store.append_run(suite_id, run_data)

                await self._notify({
                    "type": "run_complete",
                    "data": {
                        "suite_id":    suite_id,
                        "run_index":   idx,
                        "run":         run_data,
                    },
                })

            # Compute winners across all runs of this suite
            suite = self._store.get_suite(suite_id) or {}
            winners = self._compute_winners(suite.get("runs", []))
            self._store.finish_suite(suite_id, winners=winners, status="complete")

            # Mark not-running BEFORE notifying so consumers of is_running
            # (e.g. header builders) see the post-suite state. Without this,
            # the rack header stays stuck on "Stop Suite / Complete" until
            # the next message arrives.
            self.is_running = False

            await self._notify({
                "type": "suite_complete",
                "data": {
                    "suite_id":     suite_id,
                    "winners":      winners,
                    "duration_sec": int(time.perf_counter() - t_start),
                    "total_runs":   len(specs),
                },
            })

        except Exception as exc:
            log.exception("Suite failed")
            self._store.finish_suite(suite_id, winners={}, status="error")
            self.is_running = False
            await self._notify({
                "type": "suite_aborted",
                "data": {"suite_id": suite_id, "error": str(exc)},
            })
        finally:
            # Belt-and-suspenders: ensure flags are reset even if the
            # try-block returned early. is_running is set above on the
            # success and abort paths; this catches anything else.
            self.is_running = False
            self.current_id = None
            self._abort     = None
            self._task      = None

    # -------------------------------------------------------------------
    # Execute one run
    # -------------------------------------------------------------------

    async def _execute_run(self, spec: RunSpec) -> dict:
        prompt_text = prompts.get(spec.prompt_name)

        async def log_cb_sync(line: str) -> None:
            await self._notify({"type": "log", "data": {"line": line}})

        # adapt sync callback to async fire-and-forget
        def sync_log(line: str) -> None:
            asyncio.create_task(log_cb_sync(line))

        runner = LlamaRunner(
            llama_bin  = self._bin,
            bench_port = self._port,
            log_cb     = sync_log,
        )

        parallel_requests = spec.params.get("parallel", 1) if spec.profile == "throughput" else 1

        outcome: RunOutcome = await runner.run(
            model_path        = spec.model_path,
            params            = spec.params,
            prompt            = prompt_text,
            n_predict         = self._n_predict,
            n_samples         = self._runs_per_combo,
            do_warmup         = True,
            parallel_requests = parallel_requests,
            abort_event       = self._abort,
        )

        if not outcome.ok:
            return {
                "model":       spec.model_path,
                "profile":     spec.profile,
                "label":       spec.label,
                "params":      spec.params,
                "prompt_name": spec.prompt_name,
                "samples":     [],
                "aggregate":   {},
                "status":      "failed",
                "error":       outcome.error or "unknown error",
                "server_tail": outcome.server_stdout_tail,
            }

        # Aggregate
        samples_dicts = [s.as_dict() for s in outcome.samples]
        if spec.profile == "single":
            agg: Aggregate = median_of(outcome.samples) or Aggregate(0, 0, 0, 0)
            agg_dict = agg.as_dict()
        else:
            agg_dict = aggregate_parallel(outcome.samples, outcome.wall_seconds)

        return {
            "model":        spec.model_path,
            "profile":      spec.profile,
            "label":        spec.label,
            "params":       spec.params,
            "prompt_name":  spec.prompt_name,
            "samples":      samples_dicts,
            "aggregate":    agg_dict,
            "status":       "ok",
            "error":        None,
            "wall_seconds": round(outcome.wall_seconds, 2),
        }

    # -------------------------------------------------------------------
    # Winners
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_winners(runs: list[dict]) -> dict:
        """For each (model, profile), pick the best run.

        Single:     max decode_tok_s (tiebreak: min ttft_ms)
        Throughput: max aggregate_tok_s
        """
        winners: dict = {}

        # Group by (model, profile)
        groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
        for idx, r in enumerate(runs):
            if r.get("status") != "ok":
                continue
            key = (r["model"], r["profile"])
            groups.setdefault(key, []).append((idx, r))

        for (model, profile), items in groups.items():
            if profile == "single":
                def score(item):
                    _, r = item
                    a = r.get("aggregate", {})
                    return (a.get("decode_tok_s", 0), -a.get("ttft_ms", 1e9))
                best = max(items, key=score)
            else:
                def score(item):
                    _, r = item
                    return r.get("aggregate", {}).get("aggregate_tok_s", 0)
                best = max(items, key=score)

            idx, r = best
            winners.setdefault(model, {})[profile] = {
                "run_index": idx,
                "label":     r["label"],
                "params":    r["params"],
                "aggregate": r["aggregate"],
                "command":   _build_command_string(r["params"], r["model"]),
            }
        return winners


def _build_command_string(params: dict, model_path: str, port: int = 1234) -> str:
    """Build a copy-pastable llama-server command for the user."""
    parts = [
        "llama-server",
        "--host 0.0.0.0",
        f"--port {port}",
        f'-m "{model_path}"',
        f'--ctx-size {params.get("ctx_size", 8192)}',
        f'--n-gpu-layers {params.get("n_gpu_layers", 99)}',
        f'--parallel {params.get("parallel", 1)}',
        f'--threads {params.get("threads", 8)}',
        f'--batch-size {params.get("batch_size", 2048)}',
        f'--ubatch-size {params.get("ubatch_size", 512)}',
        "--metrics",
    ]
    if params.get("cache_reuse"):
        parts.append(f'--cache-reuse {params["cache_reuse"]}')
    if params.get("flash_attn"):
        parts.append("--flash-attn on")
    if params.get("cont_batching"):
        parts.append("--cont-batching")
    if params.get("draft_model"):
        parts.append(f'-md "{params["draft_model"]}"')
    return " \\\n  ".join(parts)
