"""llama_runner.py — Lifecycle of one llama-server instance for one run.

A "run" is one (model, params) combination. Within a run we collect multiple
samples (typically 3 measure + 1 warmup) and the caller aggregates them.

Server lifecycle per run:
  1. Kill any stale process on bench_port
  2. Start llama-server with params
  3. Wait until /health returns 200
  4. Send warmup request (discarded)
  5. Send N measure requests, collect Sample for each
  6. Terminate the server (SIGTERM → SIGKILL after timeout)

We restart the server between runs to ensure independence. Cache_prompt is
explicitly set to false on every request so each measurement is from a cold
KV cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import aiohttp
import requests

from metrics import Sample

log = logging.getLogger(__name__)

READY_TIMEOUT_SEC  = 120
REQUEST_TIMEOUT_SEC = 180
HEALTH_POLL_SEC    = 1.0


@dataclass
class RunOutcome:
    """Result of one run (multiple samples from one server instance)."""
    samples:       list[Sample]
    wall_seconds:  float
    server_stdout_tail: list[str]
    error:         Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.samples) > 0


class LlamaRunner:
    """One-shot runner: start server → run measurements → stop server."""

    def __init__(self, llama_bin: Path, bench_port: int,
                 log_cb: Callable[[str], None] | None = None) -> None:
        self._bin    = llama_bin
        self._port   = bench_port
        self._log_cb = log_cb or (lambda line: log.info("[llama] %s", line))
        self._stdout_buffer: list[str] = []

    # -------------------------------------------------------------------

    async def run(
        self,
        model_path:    str,
        params:        dict,
        prompt:        str,
        n_predict:     int = 256,
        n_samples:     int = 3,
        do_warmup:     bool = True,
        parallel_requests: int = 1,
        abort_event:   Optional[asyncio.Event] = None,
    ) -> RunOutcome:
        """Start a server, run measurements, stop. Returns a RunOutcome.

        Args:
          model_path:        path to .gguf
          params:            dict of llama-server flags (ctx_size, parallel, etc.)
          prompt:            text prompt to send
          n_predict:         tokens to generate per request (ignore_eos=true)
          n_samples:         number of measurement samples (median taken later)
          do_warmup:         if True, one warmup request is discarded first
          parallel_requests: how many requests in parallel during measurement
                             (1 for single profile; N for throughput)
          abort_event:       asyncio.Event — if set, aborts cleanly
        """
        self._stdout_buffer = []

        self._kill_stale()
        proc = self._start_server(model_path, params)
        if not proc:
            return RunOutcome(
                samples=[], wall_seconds=0,
                server_stdout_tail=self._stdout_buffer[-20:],
                error="failed to launch llama-server",
            )

        try:
            if not self._wait_ready(proc, abort_event):
                return RunOutcome(
                    samples=[], wall_seconds=0,
                    server_stdout_tail=self._stdout_buffer[-30:],
                    error="server did not become ready within timeout",
                )

            if abort_event and abort_event.is_set():
                return RunOutcome(
                    samples=[], wall_seconds=0,
                    server_stdout_tail=self._stdout_buffer[-10:],
                    error="aborted",
                )

            # Warmup (discarded)
            if do_warmup:
                self._emit("Warmup request…")
                await self._send_one(prompt, n_predict=16)

            # Measurement samples
            samples: list[Sample] = []
            t_wall_start = time.perf_counter()

            for i in range(n_samples):
                if abort_event and abort_event.is_set():
                    break
                self._emit(f"Sample {i + 1}/{n_samples} (parallel={parallel_requests})…")

                if parallel_requests > 1:
                    batch = await self._send_parallel(prompt, n_predict, parallel_requests)
                    samples.extend(batch)
                else:
                    s = await self._send_one(prompt, n_predict)
                    if s:
                        samples.append(s)

            wall_seconds = time.perf_counter() - t_wall_start

            return RunOutcome(
                samples=samples,
                wall_seconds=wall_seconds,
                server_stdout_tail=self._stdout_buffer[-20:],
            )

        finally:
            self._stop_server(proc)

    # ===================================================================
    # Server lifecycle
    # ===================================================================

    def _build_cmd(self, model_path: str, params: dict) -> list[str]:
        cmd = [
            str(self._bin),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "-m", model_path,
            "--ctx-size",     str(params.get("ctx_size",    8192)),
            "--n-gpu-layers", str(params.get("n_gpu_layers", 99)),
            "--parallel",     str(params.get("parallel",    1)),
            "--threads",      str(params.get("threads",     8)),
            "--batch-size",   str(params.get("batch_size",  2048)),
            "--ubatch-size",  str(params.get("ubatch_size", 512)),
            "--metrics",
        ]
        if params.get("cache_reuse"):
            cmd += ["--cache-reuse", str(params["cache_reuse"])]
        if params.get("flash_attn"):
            cmd += ["--flash-attn", "on"]
        else:
            cmd += ["--flash-attn", "off"]
        if params.get("cont_batching"):
            cmd += ["--cont-batching"]

        # Speculative decoding
        draft = params.get("draft_model")
        if draft:
            cmd += ["-md", str(draft)]
            # CRITICAL: offload draft model to GPU as well. Without this,
            # the draft runs on CPU which is catastrophically slow on Apple
            # Silicon (unified memory means main+draft both want GPU). Even
            # a 1B draft on CPU caps total throughput far below CPU-less
            # main-only inference — negating the entire speedup.
            cmd += ["--gpu-layers-draft", str(params.get("n_gpu_layers_draft", 99))]
            if params.get("spec_draft_n_max"):
                cmd += ["--spec-draft-n-max", str(params["spec_draft_n_max"])]
            if params.get("spec_draft_n_min"):
                cmd += ["--spec-draft-n-min", str(params["spec_draft_n_min"])]

        # Reasoning / thinking
        if params.get("reasoning_budget") is not None:
            cmd += ["--reasoning-budget", str(params["reasoning_budget"])]
        if params.get("enable_thinking") is False:
            cmd += ["--chat-template-kwargs", '{"enable_thinking":false}']
        elif params.get("enable_thinking") is True:
            cmd += ["--chat-template-kwargs", '{"enable_thinking":true}']

        return cmd

    def _start_server(self, model_path: str, params: dict) -> Optional[subprocess.Popen]:
        cmd = self._build_cmd(model_path, params)
        self._emit(f"$ {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._bin.parent),
                start_new_session=True,  # own process group for killpg
            )
            threading.Thread(
                target=self._drain_stdout, args=(proc.stdout,), daemon=True
            ).start()
            return proc
        except Exception as exc:
            self._emit(f"ERROR starting server: {exc}")
            return None

    def _wait_ready(self, proc: subprocess.Popen,
                    abort_event: Optional[asyncio.Event]) -> bool:
        url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.time() + READY_TIMEOUT_SEC
        while time.time() < deadline:
            if abort_event and abort_event.is_set():
                return False
            if proc.poll() is not None:
                self._emit("Server process exited before ready")
                return False
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    elapsed = READY_TIMEOUT_SEC - (deadline - time.time())
                    self._emit(f"Server ready after {elapsed:.1f}s")
                    return True
            except Exception:
                pass
            time.sleep(HEALTH_POLL_SEC)
        return False

    def _stop_server(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass

    def _kill_stale(self) -> None:
        try:
            out = subprocess.run(
                ["lsof", f"-iTCP:{self._port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            for pid_str in out.splitlines():
                try:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                    self._emit(f"Killed stale pid {pid} on port {self._port}")
                except Exception:
                    pass
            if out:
                time.sleep(1.5)
        except Exception:
            pass

    def _drain_stdout(self, pipe) -> None:
        for raw in pipe:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._stdout_buffer.append(line)
                if len(self._stdout_buffer) > 500:
                    self._stdout_buffer = self._stdout_buffer[-500:]

    # ===================================================================
    # Request paths
    # ===================================================================

    async def _send_one(self, prompt: str, n_predict: int) -> Optional[Sample]:
        url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
        payload = {
            "messages":     [{"role": "user", "content": prompt}],
            "max_tokens":   n_predict,
            "temperature":  0.0,
            "cache_prompt": False,
            "stream":       False,
            "ignore_eos":   True,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        self._emit(f"HTTP {resp.status}")
                        return None
                    data = await resp.json()
            return Sample.from_response(data)
        except Exception as exc:
            self._emit(f"Request error: {exc}")
            return None

    async def _send_parallel(self, prompt: str, n_predict: int,
                              parallel: int) -> list[Sample]:
        url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
        payload = {
            "messages":     [{"role": "user", "content": prompt}],
            "max_tokens":   n_predict,
            "temperature":  0.0,
            "cache_prompt": False,
            "stream":       False,
            "ignore_eos":   True,
        }

        async def one(session: aiohttp.ClientSession) -> Optional[Sample]:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                return Sample.from_response(data)
            except Exception:
                return None

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(*[one(session) for _ in range(parallel)])
        return [r for r in results if r is not None]

    # -------------------------------------------------------------------

    def _emit(self, line: str) -> None:
        self._log_cb(line)
