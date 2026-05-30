"""server.py — PylonRack slot for model-calibrate.

Protocol:
  - manifest / ping / pong: standard PylonRack
  - control_data: no dropdowns in header; this returns empty
  - action with control_id="suite_toggle": toggles start/abort
  - action with control_id="start_suite": start_suite(payload from UI)
  - action with control_id="stop_suite": abort
  - action with control_id="get_models":  return scanned GGUFs + sizes
  - action with control_id="get_resources": fresh memory + active llama slots
  - action with control_id="get_history":  list of past suites
  - action with control_id="get_suite":    fetch one suite's full data
  - action with control_id="delete_suite"
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import websockets
from aiohttp import web
from websockets import ServerConnection

import config as cfg_module
from model_scanner import GGUFModel, scan
from resources import (
    check_suite_feasibility,
    detect_active_llama_servers,
    get_memory_status,
)
from results_store import ResultsStore
from suite_runner import SuiteRunner
from parent_watchdog import watch_parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.cfg   = cfg_module.load()
        self.store = ResultsStore(self.cfg.results_path)

        # Connected WebSocket clients (rack + WebView share same WS)
        self.connections: set[ServerConnection] = set()

        # Suite runner — created lazily because notify needs self
        self.suite_runner: Optional[SuiteRunner] = None
        self.models: list[GGUFModel] = []

        # Bounded log
        self.log_lines: list[str] = []

        # UI status snapshot — pushed when state changes
        self.progress_text:  str = "Idle"
        self.eta_text:       str = ""
        self.metric_text:    str = ""

    def refresh_models(self) -> None:
        self.models = scan(self.cfg.hf_cache_path)

    def add_log(self, line: str) -> None:
        self.log_lines.append(line)
        if len(self.log_lines) > 1000:
            self.log_lines = self.log_lines[-1000:]


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _manifest(ui_port: int) -> dict:
    return {
        "type":    "manifest",
        "name":    "Model Calibrate",
        "version": "2.0",
        "heartbeat_interval": 5,
        "controls": [
            {"id": "suite_toggle",   "type": "button", "label": "Start Suite", "style": "primary",
             "tooltip": "Start a calibration suite for the selected models"},
            {"id": "progress_label", "type": "label",  "value": "Idle", "style": "default",
             "tooltip": "Currently running combination"},
            {"id": "eta_label",      "type": "label",  "value": "", "style": "default",
             "tooltip": "Estimated time remaining"},
            {"id": "metric_label",   "type": "label",  "value": "", "style": "default",
             "tooltip": "Last measurement"},
        ],
        "ui_url": f"http://localhost:{ui_port}/index.html",
        # No model manager (no GGUF downloads), no settings panel
        # (settings.json is rarely edited). Log is useful for debugging.
        "modes": ["log"],
    }


def _pong(state: AppState) -> dict:
    """Heartbeat reply. Status reflects connection health, not workload:
    while the slot is responding to pings, we are healthy regardless of
    whether a suite is currently running. The progress text goes into the
    message so the rack still surfaces what's happening."""
    running = state.suite_runner.is_running if state.suite_runner else False
    return {
        "type":    "pong",
        "status":  "running",
        "message": state.progress_text if running else "Ready",
    }


def _header_update(state: AppState) -> dict:
    running = state.suite_runner.is_running if state.suite_runner else False
    return {
        "type": "controls_update",
        "controls": [
            {
                "id":    "suite_toggle",
                "label": "Stop Suite" if running else "Start Suite",
                "style": "destructive" if running else "primary",
            },
            {"id": "progress_label", "value": state.progress_text,
             "style": "warning" if running else "default"},
            {"id": "eta_label",      "value": state.eta_text, "style": "default"},
            {"id": "metric_label",   "value": state.metric_text, "style": "default"},
        ],
    }


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

class SlotHandler:
    def __init__(self, state: AppState, ui_port: int) -> None:
        self._state = state
        self._ui_port = ui_port

    async def handle(self, ws: ServerConnection) -> None:
        log.info("Client connected from %s", ws.remote_address)
        self._state.connections.add(ws)
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._state.connections.discard(ws)
            log.info("Client disconnected")

    async def _dispatch(self, ws: ServerConnection, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        t = msg.get("type", "")

        if t == "manifest":
            await self._send(ws, _manifest(self._ui_port))
            await self._send(ws, _header_update(self._state))

        elif t == "ping":
            await self._send(ws, _pong(self._state))

        elif t == "control_data":
            # No header dropdowns; reply with empty for any control_id
            await self._send(ws, {
                "type":       "control_data",
                "control_id": msg.get("control_id", ""),
                "items":      [],
            })

        elif t == "action":
            await self._handle_action(ws, msg)

        elif t == "log_request":
            n = msg.get("lines", 50)
            skip = msg.get("skip", 0)
            log_total = len(self._state.log_lines)
            if skip > 0:
                end = max(0, log_total - skip)
                start = max(0, end - n)
                lines = self._state.log_lines[start:end]
                await self._send(ws, {
                    "type": "log_response", "lines": lines,
                    "total": log_total, "prepend": True,
                })
            else:
                lines = self._state.log_lines[-n:]
                await self._send(ws, {
                    "type": "log_response", "lines": lines,
                    "total": log_total,
                })

        elif t == "shutdown":
            pass  # nothing to gracefully stop; process exit is fine

    # -------------------------------------------------------------------
    # Action dispatch
    # -------------------------------------------------------------------

    async def _handle_action(self, ws: ServerConnection, msg: dict) -> None:
        cid     = msg.get("control_id", "")
        value   = msg.get("value")
        payload = msg.get("payload") or msg.get("settings") or {}

        if cid == "suite_toggle":
            # In the rack header, this toggles start/stop.
            # Without explicit selection, we can only stop a running one.
            sr = self._state.suite_runner
            if sr and sr.is_running:
                await sr.abort()
                await self._send_action_result(ws, "suite_toggle", True, "Stopping suite…")
            else:
                # Cannot start from rack alone — need model selection from UI
                await self._send_action_result(
                    ws, "suite_toggle", False,
                    "Open the slot UI to select models and start a suite.",
                )

        elif cid == "stop_suite":
            sr = self._state.suite_runner
            if sr and sr.is_running:
                await sr.abort()
                await self._send_action_result(ws, "stop_suite", True, "Stopping…")
            else:
                await self._send_action_result(ws, "stop_suite", False, "No suite running.")

        elif cid == "start_suite":
            await self._handle_start_suite(ws, payload)

        elif cid == "get_models":
            await self._handle_get_models(ws)

        elif cid == "get_resources":
            await self._handle_get_resources(ws, payload)

        elif cid == "get_history":
            await self._handle_get_history(ws)

        elif cid == "get_suite":
            await self._handle_get_suite(ws, payload)

        elif cid == "delete_suite":
            await self._handle_delete_suite(ws, payload)

        else:
            await self._send_action_result(ws, cid, False, f"Unknown action: {cid}")

    # -------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------

    async def _handle_get_models(self, ws: ServerConnection) -> None:
        self._state.refresh_models()
        items = [{
            "display_name": m.display_name,
            "full_path":    m.full_path,
            "size_gb":      m.size_gb,
        } for m in self._state.models]
        await self._send(ws, {
            "type":   "action_result",
            "action": "models",
            "data":   {"type": "models", "items": items},
        })

    async def _handle_get_resources(self, ws: ServerConnection, payload: dict) -> None:
        selected = payload.get("selected_models", [])
        # selected = [{"full_path": str, "size_gb": float}, ...]
        ctx     = int(payload.get("ctx_size", 32768))
        par     = int(payload.get("parallel", 4))

        if selected:
            tuples = [(s["full_path"], float(s.get("size_gb", 5.0))) for s in selected]
            feasibility = check_suite_feasibility(tuples, ctx_size=ctx, parallel=par,
                                                   min_required_gb=self._state.cfg.min_memory_gb)
        else:
            feasibility = {
                "ok":           True,
                "available_gb": get_memory_status().get("available_gb", 0.0),
                "memory":       get_memory_status(),
                "models":       [],
                "warnings":     [],
                "blockers":     [],
                "llama_slots":  detect_active_llama_servers(),
            }
        await self._send(ws, {
            "type":   "action_result",
            "action": "resources",
            "data":   {"type": "resources", **feasibility},
        })

    async def _handle_start_suite(self, ws: ServerConnection, payload: dict) -> None:
        # Payload:
        # { selected_models: [{full_path, size_gb}, ...],
        #   profiles: ["single", "throughput"],
        #   budget: "standard",
        #   mode: "auto" | "manual",
        #   draft_map: {<model_path>: <draft_path> or null, ...},
        #   manual_matrix: {...}  (optional) }
        selected  = payload.get("selected_models", [])
        profiles  = payload.get("profiles", ["single", "throughput"])
        budget    = payload.get("budget", "standard")
        mode      = payload.get("mode", "auto")
        manual    = payload.get("manual_matrix")
        draft_map = payload.get("draft_map") or {}

        if not selected:
            await self._send_action_result(ws, "start_suite", False, "No models selected.")
            return

        tuples = [(s["full_path"], float(s.get("size_gb", 5.0))) for s in selected]
        sr     = self._state.suite_runner
        if sr is None:
            sr = self._make_suite_runner()
            self._state.suite_runner = sr

        result = await sr.start(
            selected_models = tuples,
            profiles        = profiles,
            budget          = budget,
            mode            = mode,
            manual_matrix   = manual,
            draft_map       = draft_map,
        )

        await self._send(ws, {
            "type":   "action_result",
            "action": "start_suite",
            "data":   {"type": "start_suite", **result},
        })

        if result.get("ok"):
            # Reflect change in header
            self._state.progress_text = "Starting…"
            self._state.eta_text      = f"ETA {result['eta_seconds'] // 60}m {result['eta_seconds'] % 60}s"
            self._state.metric_text   = f"0 / {result['total_runs']}"
            await self._broadcast(_header_update(self._state))

    async def _handle_get_history(self, ws: ServerConnection) -> None:
        suites = self._state.store.list_suites_summary()
        await self._send(ws, {
            "type":   "action_result",
            "action": "history",
            "data":   {"type": "history", "suites": suites},
        })

    async def _handle_get_suite(self, ws: ServerConnection, payload: dict) -> None:
        suite_id = payload.get("suite_id", "")
        suite    = self._state.store.get_suite(suite_id)
        await self._send(ws, {
            "type":   "action_result",
            "action": "suite",
            "data":   {"type": "suite", "suite": suite},
        })

    async def _handle_delete_suite(self, ws: ServerConnection, payload: dict) -> None:
        suite_id = payload.get("suite_id", "")
        ok       = self._state.store.delete_suite(suite_id)
        await self._send(ws, {
            "type":   "action_result",
            "action": "delete_suite",
            "data":   {"type": "delete_suite", "suite_id": suite_id, "success": ok},
        })

    # -------------------------------------------------------------------
    # Suite runner factory + notify
    # -------------------------------------------------------------------

    def _make_suite_runner(self) -> SuiteRunner:
        async def notify(event: dict) -> None:
            # Update state for header on important events
            t = event.get("type")
            data = event.get("data", {})

            if t == "log":
                line = data.get("line", "")
                if line:
                    self._state.add_log(line)
                # logs only flow to subscribers (avoid spam from headers)
                await self._broadcast({
                    "type":  "log_response",
                    "lines": [line],
                    "total": -1,
                })
                return

            if t == "suite_started":
                self._state.progress_text = "Running…"
                self._state.metric_text   = f"0 / {data.get('total_runs', '?')}"
                eta = data.get("eta_seconds", 0)
                self._state.eta_text = f"ETA {eta // 60}m {eta % 60}s"

            elif t == "suite_progress":
                idx = data.get("run_index", 0)
                tot = data.get("total_runs", 0)
                current = data.get("current", {})
                self._state.progress_text = f"{current.get('label', '?')}"
                self._state.metric_text   = f"{idx + 1} / {tot}"

            elif t == "run_complete":
                run = data.get("run", {})
                agg = run.get("aggregate", {})
                if run.get("profile") == "single":
                    self._state.metric_text = (
                        f"{agg.get('decode_tok_s', 0):.0f} t/s · "
                        f"TTFT {agg.get('ttft_ms', 0):.0f}ms"
                    )
                else:
                    self._state.metric_text = (
                        f"{agg.get('aggregate_tok_s', 0):.0f} t/s agg · "
                        f"par={run['params'].get('parallel', '?')}"
                    )

            elif t == "suite_complete":
                self._state.progress_text = "Complete"
                self._state.metric_text   = f"{data.get('total_runs', '?')} runs"
                self._state.eta_text      = ""

            elif t == "suite_aborted":
                self._state.progress_text = "Aborted"
                self._state.eta_text      = ""

            await self._broadcast(_header_update(self._state))
            # Always forward to UI subscribers as well
            await self._broadcast({"type": "action_result", "action": "suite_event", "data": event})

        return SuiteRunner(
            llama_bin      = self._state.cfg.bin_path,
            bench_port     = self._state.cfg.bench_port,
            store          = self._state.store,
            notify         = notify,
            n_predict      = self._state.cfg.n_predict,
            runs_per_combo = self._state.cfg.runs_per_combo,
        )

    # -------------------------------------------------------------------
    # Send helpers
    # -------------------------------------------------------------------

    async def _send(self, ws: ServerConnection, data: dict) -> None:
        try:
            await ws.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _broadcast(self, data: dict) -> None:
        msg = json.dumps(data)
        dead = []
        for ws in self._state.connections:
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self._state.connections.discard(ws)

    async def _send_action_result(self, ws: ServerConnection,
                                   action: str, success: bool, message: str) -> None:
        await self._send(ws, {
            "type":       "action_result",
            "action":     action,
            "control_id": action,
            "success":    success,
            "message":    message,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    rack_json = Path(__file__).parent / "rack.json"
    manifest  = json.loads(rack_json.read_text())
    ws_port   = int(manifest.get("port", 8767))
    port_env  = __import__("os").environ.get("PYLON_PORT")
    if port_env:
        try:
            ws_port = int(port_env)
        except ValueError:
            pass

    # HTTP UI server on ws_port + 100 (e.g. 8867 if ws is 8767)
    ui_port = ws_port + 100

    state = AppState()
    state.refresh_models()

    handler = SlotHandler(state, ui_port)

    # ---- HTTP server for static UI ----
    static_dir = Path(__file__).parent / "static"
    app = web.Application()

    async def _root(_req):
        raise web.HTTPFound("/index.html")
    async def _config(_req):
        return web.json_response({"ws_port": ws_port})

    app.router.add_get("/",       _root)
    app.router.add_get("/config", _config)
    app.router.add_static("/",    path=str(static_dir), show_index=False)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", ui_port)
    await site.start()
    log.info("HTTP UI server on http://localhost:%d", ui_port)

    # Self-terminate if rack process dies — prevents orphan processes that
    # would hold the port open and block the next rack launch.
    asyncio.create_task(watch_parent())

    log.info("PylonRack model-calibrate slot starting on ws://localhost:%d", ws_port)
    async with websockets.serve(handler.handle, "localhost", ws_port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
