# pylonrack-calibrate

PylonRack slot for **automated calibration** of `llama-server` parameters. Given a list of local GGUF models, runs a parameter sweep to find the best `llama-server` configuration for each model, in two distinct profiles:

- **Single-use** (chat): optimizes decode tok/s and TTFT (time-to-first-token)
- **Throughput** (parallel pipeline): optimizes aggregate tok/s across N parallel slots

Persists results as a history of *suites* and surfaces a **winner** per (model, profile) — with a copy-pastable `llama-server` command.

---

## How it works

1. You select models from your HuggingFace cache (multi-select)
2. You pick profiles (single, throughput, or both)
3. You pick a budget (Quick / Standard / Thorough — controls sweep depth)
4. The slot runs each parameter combination sequentially:
   - Starts a temporary `llama-server` on `bench_port`
   - Sends a warmup request (discarded)
   - Sends 3 measurement requests with `ignore_eos: true` and `cache_prompt: false`
   - Records the `timings` object from each response (authoritative server-side measurement)
   - Stops the server
   - Aggregates samples via median
5. After all runs, winners are picked: max decode tok/s for single, max aggregate tok/s for throughput

---

## Interpreting results

Results are persisted to `~/.pylonrack/calibrate_results.json`. Each suite contains a list of runs plus a `winners` map keyed by `(model_path, profile)`. The UI surfaces winners as cards at the bottom of the Live Run tab with a **Copy command** button.

### Metrics by profile

**Single profile** (one request at a time):

| Metric | Meaning | Better |
|---|---|---|
| `decode_tok_s` | Tokens per second during generation | Higher |
| `prefill_tok_s` | Tokens per second while ingesting the prompt | Higher |
| `ttft_ms` | Time to first token (ms) | Lower |

**Throughput profile** (N requests in parallel):

| Metric | Meaning | Better |
|---|---|---|
| `aggregate_tok_s` | Total tokens generated / wall seconds | Higher |
| `per_request_decode` | Mean decode tok/s per individual request | Higher |
| `median_ttft_ms` | Median TTFT across the parallel requests | Lower |

### Choosing a profile to consult

The two profiles produce DIFFERENT winning parameters for the same model.
Which one to read depends on how the model will be invoked:

- Requests issued **one at a time** → read the single-profile winner
- Requests issued **in parallel batches** → read the throughput winner

For a use case that does both, run with both profiles enabled — the suite
produces two winners per model, one per profile.

### Speculative decoding

A run label ending in `draft=on` indicates speculative decoding was active.
The winning command will contain `-md <draft_path> --gpu-layers-draft 99`.
Whether speculative actually helps depends on draft acceptance ratio (visible
in server logs as `n_drafted` vs `n_accepted`) and tokenizer compatibility
between main and draft. The UI lists draft candidates by size but does NOT
verify tokenizer compatibility — mismatches surface at runtime as llama-server
errors. Always validate speculative vs non-speculative on the same hardware.

---

## Requirements

- macOS 14+ on Apple Silicon
- Python 3.11+
- A working [llama.cpp](https://github.com/ggerganov/llama.cpp) build (`llama-server` binary at `b9415` or later)
- HuggingFace cache with `.gguf` model files
- [PylonRack](https://github.com/marianvid/pylonrack) installed

---

## Installation

```
git clone https://github.com/marianvid/pylonrack-calibrate
cd pylonrack-calibrate
```

Dependencies install automatically on first launch via `start.sh`.

### Configure `settings.json`

```json
{
  "llama_bin":      "/path/to/llama.cpp/build/bin/llama-server",
  "hf_cache":       "/path/to/HuggingFace/hub",
  "bench_port":     1235,
  "results_file":   "~/.pylonrack/calibrate_results.json",
  "log_file":       "~/.pylonrack/calibrate.log",
  "n_predict":      256,
  "runs_per_combo": 3,
  "min_memory_gb":  6.0
}
```

| Key | Description |
|-----|-------------|
| `llama_bin` | Absolute path to compiled `llama-server` binary |
| `hf_cache` | Absolute path to your HuggingFace hub cache |
| `bench_port` | Port used by the temporary calibration `llama-server` (separate from your main instance) |
| `n_predict` | Tokens to generate per sample (with `ignore_eos`, this is exact) |
| `runs_per_combo` | Number of measurement samples per parameter combination (median taken) |
| `min_memory_gb` | Refuse to start a suite if available memory is below this |

`settings.json` is gitignored.

---

## Adding to PylonRack

1. Open PylonRack
2. Click `+` and browse to this folder
3. Press ▶ to activate the slot

---

## Header controls

| Control | Type | Description |
|---|---|---|
| Start Suite | Button | Toggles between Start (when idle) and Stop Suite (while running) |
| Progress | Label | Currently running model + parameters, or status |
| ETA | Label | Estimated time remaining |
| Metric | Label | Last measurement value (decode tok/s · TTFT, or aggregate tok/s) |

The slot exposes a `ui_url` — the main UI lives in the WebView panel, not in the header.

---

## WebView UI

Three tabs:

### Setup
- Models list with per-model size, fit status (ok / tight / no fit) relative to current available memory
- **Per-model draft picker** — when a model is selected AND the Single profile is active, a dropdown appears under each model listing smaller models in the cache (≤50% of main size) that can serve as a draft for speculative decoding. Only applied to single-profile runs; throughput runs always skip the draft.
- Profile cards (toggleable): Single-use / Throughput
- Budget radios: Quick (~2 combos/profile) / Standard (~4-5) / Thorough (~7-8)
- Mode toggle: Auto sweep (default) — Manual matrix flag is pending removal
- Start Suite button + ETA preview

### Live Run
- Suite ID + current status + elapsed/ETA counters + progress bar
- Live table of runs (one row per parameter combination, plus a placeholder for the running one)
- Winners cards once the suite completes

### History
- List of all past suites with duration, run count, profiles tested
- Click a row to expand: full winners + runs table for that suite
- Delete button per suite

---

## Fixed prompts

Each suite uses three fixed prompts of different lengths to characterize prefill behaviour:

- **SHORT** (~32 tokens) — chat scale, used for throughput profile
- **MEDIUM** (~440 tokens) — single-article scale, used for single profile
- **LONG** (~3660 tokens) — long-context scale, reserved for future use

Content is intentionally generic — calibration measures the engine, not the response quality. Only the token count and structure matter.

---

## Auto sweep strategy

### Single-use
| Budget | Combinations |
|---|---|
| Quick | ub=512, ub=2048 |
| Standard | ub=512, ub=1024, ub=2048, plus one larger ctx |
| Thorough | grid over ub × ctx × flash_attn |

Draft model, when picked, is applied to every single-profile combination.

### Throughput
| Budget | Combinations |
|---|---|
| Quick | parallel=4, parallel=8 |
| Standard | parallel=2/4/8/16 + best parallel × bigger batch |
| Thorough | full grid over parallel × batch/ubatch |

Throughput runs never use a draft model (parallel slots already saturate the
GPU; drafts would regress aggregate throughput).

---

## Resource check

Before starting a suite, the slot checks available memory via `vm_stat`:
- Memory accounting: `free + inactive + purgeable + speculative`
- Per-model fit estimate: weights + KV-cache + 2 GB safety margin
- Detects active `pylonrack-llama` slots and warns

If any selected model can't fit, the suite refuses to start.

---

## File structure

```
pylonrack-calibrate/
├── rack.json              ← PylonRack slot manifest
├── settings.json          ← local configuration (gitignored)
├── start.sh               ← venv bootstrap + launch
├── server.py              ← WebSocket + HTTP server, AppState, dispatch
├── config.py              ← AppConfig
├── prompts.py             ← three fixed prompts (short/medium/long)
├── metrics.py             ← Sample + Aggregate extraction from `timings`
├── resources.py           ← memory check + pylonrack-llama detection
├── sweep_strategy.py      ← auto-sweep / manual-matrix run spec builder
├── llama_runner.py        ← one llama-server lifecycle per run
├── suite_runner.py        ← orchestrates a full suite
├── results_store.py       ← schema v2 JSON persistence
├── model_scanner.py       ← scans HF cache for .gguf
├── requirements.txt
├── static/                ← WebView UI
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
└── tests/
    ├── test_e2e.py        ← backend smoke test (no WS)
    └── test_e2e_ws.py     ← full WebSocket flow test
```

---

## Results schema

`~/.pylonrack/calibrate_results.json`:

```json
{
  "version": 2,
  "suites": [
    {
      "id":            "suite_YYYYMMDD_HHMMSS",
      "started_at":    "ISO timestamp",
      "duration_sec":  int,
      "budget":        "quick|standard|thorough",
      "mode":          "auto|manual",
      "profiles":      ["single", "throughput"],
      "models_tested": [path, ...],
      "runs": [
        {
          "model":       path,
          "profile":     "single|throughput",
          "label":       "ub=2048, ctx=8192",
          "params":      {...},
          "prompt_name": "medium",
          "samples":     [...],
          "aggregate":   {...},
          "status":      "ok|failed",
          "error":       null
        }
      ],
      "winners": {
        "<model_path>": {
          "single":     {"label", "params", "aggregate", "command"},
          "throughput": {...}
        }
      }
    }
  ]
}
```

---

## License

MIT — use freely, no warranty.
