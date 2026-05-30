# AGENTS.md — pylonrack-calibrate

> Machine-readable implementation reference. Not human documentation.
> Audience: AI agents continuing implementation.
> Audit: append knowledge after each implementation session.

---

## SYSTEM_IDENTITY

```
name: pylonrack-calibrate
type: PylonRack slot application
language: Python 3.11+
repo: github.com/marianvid/pylonrack-calibrate
local_path: /Volumes/Marian_Backup/work/pylonrack-slots/calibrate/
rack_protocol: PylonRack WebSocket protocol (see pylonrack/AGENTS.md)
ws_port: 8767 (from rack.json, overridable via PYLON_PORT env var)
ui_port: ws_port + 100 (HTTP static server for the WebView UI)
venv: .venv/ (auto-created by start.sh on first run)
deps: websockets>=12.0, aiohttp>=3.9, psutil>=5.9, requests>=2.31
```

---

## PURPOSE

Automated calibration of `llama-server` parameters for **ParallaxVox pipeline use**.
Not a general-purpose benchmark — designed to find optimal flags for the specific
workload patterns of ParallaxVox (geopolitical news monitoring + analysis):

- **Single profile**: ad-hoc analysis, manual queries, chat-style single requests
- **Throughput profile**: pipeline runs (RSS ingestion → ideas extraction → consolidation → article generation)

The authoritative output is a **winner** per (model, profile) tuple: optimal
`llama-server` command line that the user copy-pastes into their pipeline
configuration. ParallaxVox itself does NOT read calibrate's results — the
human is the bridge.

---

## MODULE_MAP

```
server.py          — WebSocket+HTTP entry, AppState, SlotHandler, dispatch
config.py          — AppConfig + ServerConfig dataclasses, settings.json loader
prompts.py         — three fixed prompts: SHORT (~32), MEDIUM (~440), LONG (~3660)
metrics.py         — Sample + Aggregate extraction from llama-server timings
resources.py       — vm_stat memory check + pylonrack-llama slot detection
sweep_strategy.py  — RunSpec generator: build_auto_sweep + build_manual_matrix
llama_runner.py    — LlamaRunner: one llama-server lifecycle per run
suite_runner.py    — SuiteRunner: orchestrates a full suite (notify callback)
results_store.py   — schema v2 JSON persistence
model_scanner.py   — scan HF cache for .gguf files; dedup symlinks; filter obsolete quants
parent_watchdog.py — IDENTICAL copy in every slot (self-terminate on rack death)
static/            — WebView UI (HTML + CSS + JS, served via aiohttp)
settings.json      — user config (gitignored)
rack.json          — PylonRack slot manifest
start.sh           — venv bootstrap + exec python3 server.py
tests/             — pytest test suite
```

---

## CONFIG_SCHEMA

### settings.json
```json
{
  "llama_bin":      "/Volumes/Marian_Backup/git/llama.cpp/build/bin/llama-server",
  "hf_cache":       "/Volumes/Marian_Backup/HF_Cache/hub",
  "bench_port":     1235,
  "results_file":   "~/.pylonrack/calibrate_results.json",
  "log_file":       "~/.pylonrack/calibrate.log",
  "n_predict":      256,
  "runs_per_combo": 3,
  "min_memory_gb":  6.0
}
```

- `bench_port` MUST differ from the user's main llama-server (typically 1234).
  We start/stop a separate llama-server per run.
- `n_predict` is exact when paired with `ignore_eos:true` (default in our requests).
- `runs_per_combo`: 3 is enough for median-of-3 to be stable; higher is overkill.
- `min_memory_gb`: refuse to start suite if `available_gb < min_memory_gb`. Set
  to 6 GB which leaves headroom even with Servoy VM running concurrently.

### rack.json
```json
{ "name": "Model Calibrate", "version": "2.0", "start": "./start.sh",
  "port": 8767, "startup_delay": 0 }
```

---

## ARCHITECTURE_NOTES

### Why a separate llama-server per run

Calibration measures cold-start performance. Reusing a process across runs
would keep KV cache warm and contaminate `prefill_tok_s` measurements. Cost:
~15s startup + ~3s shutdown per run. Acceptable given run wall-time of ~30s.

### Why three fixed prompts and not user-provided

Reproducibility. Different prompt content produces different prefill
performance even at identical token counts (tokenization variance). Fixed
prompts mean: same run two days apart, on the same model, should produce
results within noise tolerance. Prompts are intentionally generic — quality
of output is not measured, only engine performance.

Token counts (approximate, family-dependent):
- SHORT: ~32 tokens → throughput profile (lots of small concurrent requests)
- MEDIUM: ~440 tokens → single profile (single-article-scale chat)
- LONG: ~3660 tokens → reserved for future long-context profile (not used yet)

### Authoritative metrics source

We read `timings` from llama-server's response JSON, NOT wall-clock-based
estimates. Server-side measurement excludes network latency and serialization
overhead. Fields used:
- `prompt_n`, `prompt_per_second` → prefill tok/s
- `predicted_n`, `predicted_per_second` → decode tok/s
- `prompt_ms` first sample → TTFT estimate
- For speculative decoding: `draft_n`, `draft_n_accepted` → acceptance ratio

We set `cache_prompt:false` on every request to force cold KV cache. We set
`ignore_eos:true` so n_predict is exact (otherwise the model may stop early
and skew tok/s averages).

### Aggregation rules

**Single profile**: median of 3 samples on each metric independently. Decode
speed is stable; TTFT can have outliers, median is robust.

**Throughput profile**: aggregate_tok_s = sum of all parallel samples' tokens
/ wall-clock seconds. This is the true throughput a pipeline would see. NOT
the sum of per-request tok/s (which would over-count).

Winners chosen via:
- Single: max `decode_tok_s` (tiebreak: min `ttft_ms`)
- Throughput: max `aggregate_tok_s`

### Auto sweep strategy

Not a full grid search. We hand-picked combinations that cover the practical
decision space without combinatorial explosion:

**Single budget=standard**: ub=512, ub=1024, ub=2048, ub=2048 at ctx=32768.
This answers: "what ubatch size and what ctx fits my use case best?"

**Throughput budget=standard**: parallel=2/4/8/16, plus best parallel at
larger batch_size. This answers: "how many concurrent slots before
throughput plateaus?"

For PVx pipeline, standard budget is the recommended default. Quick is for
smoke-testing a new model. Thorough only when publishing comparison data.

### Speculative decoding (draft model) — single profile only

Draft is propagated into RunSpec.params via `build_auto_sweep(draft_model=)`.
The profile filter `if profile == "single"` ensures throughput runs never
use a draft (parallel slots saturate GPU; drafts regress throughput).

Known gotcha on Apple Silicon: draft model MUST be GPU-offloaded
(`--gpu-layers-draft 99`) or CPU bottleneck eclipses any speculative gain.
This is hardcoded in `llama_runner._build_cmd`.

### Resource check (pre-flight)

`vm_stat` provides per-page counters. Available = `free + inactive + purgeable
+ speculative`. Wired and active are not available for our purposes.

Per-model size estimate: `weights + 6% KV-cache + 2 GB safety`. The 6%
figure is heuristic for ctx=32768 at K-quant 4-bit. Real KV size scales
with ctx, but the suite ALWAYS allocates with ctx=32768 to ensure the
worst-case throughput run fits.

We also detect active pylonrack-llama slots via lsof+ps and warn the user.
A running llama instance means the calibrate-spawned one will compete for
GPU; the user can either accept the noise or stop the other slot first.

---

## PROTOCOL_INTEGRATION

Full manifest schema is documented in pylonrack/AGENTS.md. Highlights specific
to calibrate:

```python
{
  "type":    "manifest",
  "name":    "Model Calibrate",
  "version": "2.0",
  "heartbeat_interval": 5,
  "controls": [
    # suite_toggle is the only actionable button visible in the header
    {"id": "suite_toggle",   "type": "button", "label": "Start Suite", "style": "primary",
     "tooltip": "Start a calibration suite for the selected models"},
    # Three labels surface live state without taking action
    {"id": "progress_label", "type": "label", "value": "Idle", "style": "default",
     "tooltip": "Currently running combination"},
    {"id": "eta_label",      "type": "label", "value": "", "style": "default",
     "tooltip": "Estimated time remaining"},
    {"id": "metric_label",   "type": "label", "value": "", "style": "default",
     "tooltip": "Last measurement"},
  ],
  "ui_url": "http://localhost:8867/index.html",
  "modes": ["log"],   # only Log mode button appears in rack header
}
```

`modes: ["log"]` is critical — without it, rack would show Models and Settings
buttons too (defaults for slot-with-manifest behavior). Calibrate has no
GGUF download or settings panel; only log access is meaningful.

### pong response

Status is always `running` while the WebSocket is responsive, regardless of
whether a suite is executing. The progress text goes in `message`. This was
changed 2026-05-30 after a bug where status=warning during a running suite
caused the rack to hide the WebView (because rack rendering treated warning
as "reachable-but-degraded" and showed a placeholder instead of the UI).

### Action contract

Client → server actions used:
- `start_suite` (payload: selected_models, profiles, budget, mode, draft_map, manual_matrix?)
- `stop_suite` (no payload)
- `get_models`, `get_resources` (payload for live fit calc), `get_history`,
  `get_suite` (payload: suite_id), `delete_suite` (payload: suite_id)

Server → client events during a suite (broadcast):
- `suite_started`, `suite_progress`, `run_complete`, `suite_complete`,
  `suite_aborted` — all wrapped in `action_result` with `action: suite_event`.
- `log` events for streaming detail into the log panel.

Header updates flow as `controls_update` messages built by `_header_update()`.
The critical contract: `is_running` flag in SuiteRunner MUST be set to False
BEFORE the final `notify({"type": "suite_complete"})` call, so the header
builder sees the post-suite state. See HISTORY for the bug.

---

## PARALLAXVOX_INTEGRATION

The slot is NOT auto-wired into ParallaxVox. Outputs are JSON in
`~/.pylonrack/calibrate_results.json`; the human reads winners and decides
which settings to copy.

### Pipeline-stage to calibrate-profile mapping

| Pipeline stage             | Workload pattern               | Profile     | Winner relevance               |
|----------------------------|--------------------------------|-------------|--------------------------------|
| RSS → Ideas extraction     | Batch, many short prompts      | throughput  | aggregate_tok_s primary        |
| Ideas → Consolidation      | A few parallel groups          | throughput  | aggregate_tok_s, par=2-4       |
| Consolidation → Articles   | Long generation, may parallel  | throughput  | per_request_decode + aggregate |
| Ad-hoc analysis            | Single chat-style query        | single      | decode_tok_s + TTFT            |

For each ParallaxVox model role (ideas/consolidation/articles), the user
should:
1. Run calibrate suite with that model + throughput profile + standard budget
2. Inspect the throughput winner's parameters
3. Update ParallaxVox config (or the pylonrack-llama slot if used live) with
   the winning `ctx_size`, `parallel`, `batch_size`, `ubatch_size`,
   `flash_attn`, `cont_batching` values
4. Validate by running a real pipeline pass and comparing observed tok/s
   against the calibrated number

### Models in scope (as of 2026-05-30)

- `Llama 3.1 8B Instruct` (Q4_K_M, ~4.6 GB) — ideas extraction
- `Gemma 4 26B-A4B-it` (Q4_K_M, ~17 GB) — consolidation
- `Qwen 3.6 35B-A3B` (Q4_K_M, ~21 GB) — articles
- `Qwen 3.5 4B` (Q4_K_M, ~2.8 GB) — small/screening tasks
- `Llama 3.2 1B Instruct` (Q4_K_M, ~0.8 GB) — draft model for Llama 8B speculative

Draft model is currently set up only for Llama 3.1 8B → Llama 3.2 1B. Qwen
draft pairing has been tested and does NOT work (tokenizer mismatch between
Qwen 3.6 35B and Qwen 3.5 4B). Don't waste a suite on it.

Cloud APIs (Claude, GPT) are used for the highest-quality articles where
local inference is not good enough. Those don't go through calibrate.

### Hardware envelope

Apple M3 Max, 128 GB unified memory, 16 CPU cores (12P+4E), 40 GPU cores.
Servoy VM runs concurrently using ~50 GB. Effective ParallaxVox envelope:
~70 GB. Calibrate enforces `min_memory_gb: 6` and refuses a suite if any
selected model would push over the limit.

---

## PITFALLS

### Obsolete GGUF quants

llama.cpp build b4282+ rejects pre-packed `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`
files (the new online-repacking path requires plain Q4_0). model_scanner.py
FILTERS these out. If you see a model in HF cache that doesn't appear in
calibrate UI, check whether the filename contains one of those suffixes.

### Symlink duplicates

HF cache stores blobs by hash and snapshots by tag. If a model is downloaded
twice via different tools (e.g. `huggingface-cli download FILE` vs the
automatic snapshot path), you can end up with two physical files of identical
content. model_scanner.py deduplicates via `os.path.realpath()`. Earlier
versions did not — a duplicate Qwen 3.5 4B (3 GB wasted) was found and
manually cleaned up on 2026-05-30.

### WKWebView dialog suppression

The rack's WKWebView does not implement WKUIDelegate. As a result, `alert()`,
`confirm()`, and `prompt()` from the slot's WebView UI silently no-op. The
calibrate UI uses:
- Two-step "arm" pattern for destructive actions (delete suite, stop suite)
- A custom toast notification system (`showToast()` in app.js) for info/error
  messages

Do NOT add `confirm()` / `alert()` calls. They will appear to work in a
browser but break in the rack.

### Suite state ordering bug (fixed 2026-05-30)

SuiteRunner._run_loop must set `is_running = False` BEFORE emitting the final
`suite_complete` event, not in the `finally` block. Otherwise the header
builder reads stale `is_running=True` and the "Start Suite" button stays
stuck as "Stop Suite / Complete". See suite_runner.py — both `suite_complete`
and `suite_aborted` paths set the flag manually before notifying; the
`finally` is now belt-and-suspenders only.

### Draft model speed regression on Apple Silicon

Without `--gpu-layers-draft 99`, the draft model runs on CPU, which is
catastrophically slow on M-series chips. The main model on GPU + draft on
CPU produces decode rates SLOWER than the main model alone. Fixed in
`llama_runner._build_cmd` — `--gpu-layers-draft 99` is always passed when
`-md` is. Acceptance ratios of 60-70% with the fix yield ~50% speedup on
Llama 8B + Llama 1B draft pairing.

### Server-did-not-become-ready timeout

Common cause: model file is corrupt, wrong quant format, or path is wrong.
Look at `server_stdout_tail` in the failed run's record — llama-server's
actual error appears there. Old code attempted to load Q4_0_8_8 silently;
now filtered.

Less common: memory pressure from another process spawned llama-server got
OOM-killed before becoming ready. Free memory and retry.

---

## DESIGN_DECISIONS

### Why no benchmarking with realistic prompts from ParallaxVox

Tempting but rejected. Two reasons:
1. Reproducibility — content varies day-to-day; calibrate would never
   produce comparable numbers across runs
2. Separation of concerns — engine performance and content quality are
   orthogonal; mixing them obscures both

The three fixed prompts (short/medium/long token counts) cover the
performance envelope. Real-world variance comes from prompt content shape
(reasoning patterns, vocabulary distribution) but calibrate measures the
floor, not the typical case.

### Why no auto-config-write to pylonrack-llama settings.json

Two reasons:
1. The user runs calibrate occasionally, but reads winners many times
   afterward (referring back). Auto-applying would lose that decoupling.
2. Different model roles use different settings — the user picks per
   ParallaxVox stage. Auto-apply would require role-aware mapping that
   doesn't exist in either slot's data model.

User copies the winning command. Friction is intentional.

### Why manual matrix mode exists (and will be removed)

Original plan: power-users could specify arbitrary axis × value grids. For
ParallaxVox use it's overkill — the auto sweep covers the same decision
space with curated combinations. The UI currently shows it as "coming
soon". Decision pending — flagged for removal when ParallaxVox calibration
workflow is fully validated. (Owner: Marian)

---

## HISTORY

```
2026-05-30 Slot built from scratch. Core features:
           - Auto sweep generator (single, throughput, three budgets)
           - LlamaRunner with start/stop per run
           - SuiteRunner with notify callback for progress/winners/log
           - ResultsStore schema v2 (incremental persist, abort-safe)
           - vm_stat-based resource check + pylonrack-llama detection
           - WebView UI (Setup / Live / History tabs) with green parallaxvox theme
           - WebSocket + aiohttp HTTP servers on adjacent ports
2026-05-30 Bug fixes during initial integration:
           - model_scanner: filter Q4_0_X_X obsolete formats, dedup symlinks
           - suite_runner: is_running set to False BEFORE notify(suite_complete)
           - server._pong: status always "running" while suite executes
             (previously "warning" — caused rack to hide WebView mid-run)
           - llama_runner: --gpu-layers-draft 99 always passed with -md
           - JS: two-step arm pattern for delete/stop (WKWebView confirm() broken)
           - JS: showToast() replaces alert() for the same reason
2026-05-30 Speculative decoding support:
           - sweep_strategy.build_auto_sweep accepts draft_model parameter
           - server.start_suite accepts draft_map: {model_path: draft_path}
           - suite_runner.start propagates draft_map into per-model sweeps
           - UI: per-model draft picker visible only when single profile active
           - Draft applied to single profile only (throughput intentionally skipped)
2026-05-30 Documentation + git commit baseline established for handoff
```

---

## TEST_COMMANDS

```sh
# E2E backend smoke (no WebSocket)
.venv/bin/python -m pytest tests/test_e2e.py -v

# Full WebSocket flow (starts server on a random port, drives via WS client)
.venv/bin/python -m pytest tests/test_e2e_ws.py -v

# Manual: run server standalone
PYLON_PORT=8767 .venv/bin/python server.py

# Manual: inspect results store
python3 -c "import json,os; d=json.load(open(os.path.expanduser('~/.pylonrack/calibrate_results.json'))); print(len(d['suites']),'suites')"
```

---

## EXTENSION_POINTS

When adding features, anchor them here:

### Adding a new profile (beyond single + throughput)

1. Add the profile name to allowed values in server `_handle_start_suite`
   (or remove the implicit allow-list)
2. Add `_xxx_combos(budget)` builder in `sweep_strategy.py`
3. Extend `build_auto_sweep` to dispatch on the new profile name
4. Pick a prompt (`prompts.SHORT/MEDIUM/LONG` or add a new one)
5. Extend `_compute_winners` in `suite_runner.py` with the scoring rule
6. Add a card to the Setup tab's profile-cards section in `index.html`

### Adding a new metric

1. Extract from `timings` in `metrics.py` Sample.from_timings()
2. Add field to `Aggregate.as_dict()`
3. Add column to runs table in `app.js` renderRunRow()
4. Update `_build_command_string` in suite_runner.py if it should flow to
   the copy-pastable command

### Hooking a new model role into ParallaxVox

User-side workflow, not slot-side. Run a calibrate suite with that model in
the profile that matches the workload, copy the winner's command, and apply
the flags wherever ParallaxVox spawns llama-server (or update the
pylonrack-llama settings_map if running live).
