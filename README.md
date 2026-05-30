## ParallaxVox workflow

This slot is built for one explicit purpose: calibrating the local
`llama-server` setup used by [ParallaxVox](https://parallaxvox.com), a
geopolitical news monitoring and analysis pipeline.

The pipeline has four model-heavy stages, each with its own workload shape:

| Stage | Workload | Profile to calibrate | Models tested |
|---|---|---|---|
| RSS → Ideas | Many short prompts in batch | throughput | Llama 3.1 8B (+ draft) |
| Ideas → Consolidation | A few parallel groups | throughput (par=2-4) | Gemma 4 26B-A4B |
| Consolidation → Articles | Long generation, may be parallel | throughput | Qwen 3.6 35B-A3B |
| Ad-hoc analysis | Single chat-style query | single | Any of the above |

For each stage:

1. Run a Standard suite with the relevant model and matching profile
2. Read the winner's `Copy command` and apply those flags wherever the
   stage spawns `llama-server` (or update the pylonrack-llama settings_map
   if you run it interactively)
3. Validate by running a real pipeline pass and comparing observed tok/s
   against the calibrated number

Results are not auto-applied — the human is the bridge between calibration
and production. This is intentional: different stages use different settings
and there's no role-aware mapping in the slot's data model.

---

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

See **ParallaxVox workflow** below for the specific use case this slot was
built to support.

---

## ParallaxVox workflow

This slot is built for one explicit purpose: calibrating the local
`llama-server` setup used by [ParallaxVox](https://parallaxvox.com), a
geopolitical news monitoring and analysis pipeline.

The pipeline has four model-heavy stages, each with its own workload shape:

| Stage | Workload | Profile to calibrate | Models tested |
|---|---|---|---|
| RSS → Ideas | Many short prompts in batch | throughput | Llama 3.1 8B (+ draft) |
| Ideas → Consolidation | A few parallel groups | throughput (par=2-4) | Gemma 4 26B-A4B |
| Consolidation → Articles | Long generation, may be parallel | throughput | Qwen 3.6 35B-A3B |
| Ad-hoc analysis | Single chat-style query | single | Any of the above |

For each stage:

1. Run a Standard suite with the relevant model and matching profile
2. Read the winner's `Copy command` and apply those flags wherever the
   stage spawns `llama-server` (or update the pylonrack-llama settings_map
   if you run it interactively)
3. Validate by running a real pipeline pass and comparing observed tok/s
   against the calibrated number

Results are not auto-applied — the human is the bridge between calibration
and production. This is intentional: different stages use different settings
and there's no role-aware mapping in the slot's data model.

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
- **LONG** (~3660 tokens) — long-context / consolidation scale

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
