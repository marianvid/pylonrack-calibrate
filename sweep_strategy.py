"""sweep_strategy.py — Generate parameter combinations for calibration.

Two profiles, two strategies:

  SINGLE-USE profile (chat, batch=1):
    - Goal: maximize decode tok/s + minimize TTFT
    - Variables: ctx_size, ubatch_size, optional draft model
    - Fixed: parallel=1, flash_attn=on, n_gpu_layers=99

  THROUGHPUT profile (parallel pipeline):
    - Goal: maximize aggregate tok/s across N parallel slots
    - Variables: parallel, batch_size, ubatch_size
    - Fixed: ctx_size (sweep separately if needed), flash_attn=on,
             cont_batching=on, n_gpu_layers=99

Budget controls how deep the sweep goes:
  quick    = baseline + 1-2 obvious tweaks (~4 runs / model / profile)
  standard = baseline + main variations    (~8 runs / model / profile)
  thorough = full grid                     (~15 runs / model / profile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Run spec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One run = one (model, profile, params, prompt) combination."""
    model_path:   str
    profile:      str          # "single" | "throughput"
    params:       dict
    prompt_name:  str          # "short" | "medium" | "long"
    label:        str          # human-readable, e.g. "ub=2048, fa=on"

    def as_dict(self) -> dict:
        return {
            "model_path":  self.model_path,
            "profile":     self.profile,
            "params":      dict(self.params),
            "prompt_name": self.prompt_name,
            "label":       self.label,
        }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Hardware-reasonable defaults for M3 Max + Metal
BASE_PARAMS = {
    "n_gpu_layers": 99,
    "threads":      8,
    "flash_attn":   True,
    "cache_reuse":  256,
}

# Single-use baseline
SINGLE_BASE = {
    **BASE_PARAMS,
    "ctx_size":      8192,
    "parallel":      1,
    "batch_size":    2048,
    "ubatch_size":   512,
    "cont_batching": False,
}

# Throughput baseline
THROUGHPUT_BASE = {
    **BASE_PARAMS,
    "ctx_size":      32768,
    "parallel":      4,
    "batch_size":    2048,
    "ubatch_size":   512,
    "cont_batching": True,
}


# ---------------------------------------------------------------------------
# Strategy A — auto sweep
# ---------------------------------------------------------------------------

def _single_use_combos(budget: str) -> list[dict]:
    """Generate parameter dicts for single-use profile."""
    if budget == "quick":
        # Just two ubatch sizes
        return [
            {**SINGLE_BASE, "ubatch_size": 512},
            {**SINGLE_BASE, "ubatch_size": 2048},
        ]
    if budget == "standard":
        return [
            {**SINGLE_BASE, "ubatch_size": 512},
            {**SINGLE_BASE, "ubatch_size": 1024},
            {**SINGLE_BASE, "ubatch_size": 2048},
            # Larger context
            {**SINGLE_BASE, "ubatch_size": 2048, "ctx_size": 32768},
        ]
    # thorough
    return [
        {**SINGLE_BASE, "ubatch_size": 512},
        {**SINGLE_BASE, "ubatch_size": 1024},
        {**SINGLE_BASE, "ubatch_size": 2048},
        {**SINGLE_BASE, "ubatch_size": 4096},
        {**SINGLE_BASE, "ubatch_size": 2048, "ctx_size": 32768},
        {**SINGLE_BASE, "ubatch_size": 2048, "ctx_size": 65536},
        {**SINGLE_BASE, "ubatch_size": 2048, "flash_attn": False},
    ]


def _throughput_combos(budget: str) -> list[dict]:
    """Generate parameter dicts for throughput profile."""
    if budget == "quick":
        return [
            {**THROUGHPUT_BASE, "parallel": 4},
            {**THROUGHPUT_BASE, "parallel": 8},
        ]
    if budget == "standard":
        return [
            {**THROUGHPUT_BASE, "parallel": 2},
            {**THROUGHPUT_BASE, "parallel": 4},
            {**THROUGHPUT_BASE, "parallel": 8},
            {**THROUGHPUT_BASE, "parallel": 16},
            # Bigger batch at best parallel — replicated for combo coverage
            {**THROUGHPUT_BASE, "parallel": 8, "batch_size": 4096, "ubatch_size": 1024},
        ]
    # thorough
    return [
        {**THROUGHPUT_BASE, "parallel": 2},
        {**THROUGHPUT_BASE, "parallel": 4},
        {**THROUGHPUT_BASE, "parallel": 8},
        {**THROUGHPUT_BASE, "parallel": 16},
        {**THROUGHPUT_BASE, "parallel": 24},
        {**THROUGHPUT_BASE, "parallel": 8, "batch_size": 4096, "ubatch_size": 1024},
        {**THROUGHPUT_BASE, "parallel": 8, "batch_size": 1024, "ubatch_size": 256},
        {**THROUGHPUT_BASE, "parallel": 16, "batch_size": 4096, "ubatch_size": 1024},
    ]


def _label_for(params: dict, profile: str) -> str:
    if profile == "single":
        bits = [
            f"ctx={params['ctx_size']}",
            f"ub={params['ubatch_size']}",
        ]
        if not params.get("flash_attn"):
            bits.append("fa=off")
    else:
        bits = [
            f"par={params['parallel']}",
            f"b={params['batch_size']}",
            f"ub={params['ubatch_size']}",
        ]
    # Draft model applies to single profile only; show as "draft=on" when
    # present so the user can distinguish runs at a glance in the table.
    if params.get("draft_model"):
        bits.append("draft=on")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_auto_sweep(
    model_path: str,
    profiles: list[str],
    budget: str = "standard",
    draft_model: Optional[str] = None,
) -> list[RunSpec]:
    """Build a list of RunSpecs for one model, auto mode.

    Args:
      model_path:  GGUF path
      profiles:    subset of ["single", "throughput"]
      budget:      "quick" | "standard" | "thorough"
      draft_model: Optional path to a small GGUF used for speculative decoding.
                   Applied ONLY to single-profile runs — throughput with
                   parallel slots gains nothing from drafts and may regress.

    Returns:
      list of RunSpec, one per parameter combination.
      For single profile, prompt is "medium" (most representative).
      For throughput, prompt is "short" (fast, throughput is what matters).
    """
    specs: list[RunSpec] = []
    for profile in profiles:
        if profile == "single":
            combos = _single_use_combos(budget)
            prompt = "medium"
        elif profile == "throughput":
            combos = _throughput_combos(budget)
            prompt = "short"
        else:
            continue
        for params in combos:
            p = dict(params)
            if profile == "single" and draft_model:
                p["draft_model"] = draft_model
            specs.append(RunSpec(
                model_path  = model_path,
                profile     = profile,
                params      = p,
                prompt_name = prompt,
                label       = _label_for(p, profile),
            ))
    return specs


def build_manual_matrix(
    model_path: str,
    profile: str,
    base_params: dict,
    sweeps: dict[str, list],
    prompt_name: str = "medium",
) -> list[RunSpec]:
    """Strategy B — build sweep from explicit axis × values matrix.

    Args:
      base_params:  dict of fixed parameters
      sweeps:       {axis_name: [v1, v2, v3], ...}
                    Cartesian product is taken. Be careful — small grids only.

    Example:
      build_manual_matrix(
        model_path, "throughput",
        base_params={"ctx_size": 32768, "n_gpu_layers": 99, ...},
        sweeps={"parallel": [4, 8, 16], "ubatch_size": [512, 1024]},
      )
    """
    import itertools

    if not sweeps:
        return [RunSpec(
            model_path  = model_path,
            profile     = profile,
            params      = dict(base_params),
            prompt_name = prompt_name,
            label       = _label_for(base_params, profile),
        )]

    axis_names = list(sweeps.keys())
    value_lists = [sweeps[k] for k in axis_names]

    specs = []
    for values in itertools.product(*value_lists):
        params = dict(base_params)
        for k, v in zip(axis_names, values):
            params[k] = v
        specs.append(RunSpec(
            model_path  = model_path,
            profile     = profile,
            params      = params,
            prompt_name = prompt_name,
            label       = _label_for(params, profile),
        ))
    return specs


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate_suite_duration_sec(specs: list[RunSpec], runs_per_combo: int = 3) -> int:
    """Rough ETA in seconds for a full suite.

    Per-run cost breakdown:
      - Server startup:     ~15s
      - Server shutdown:    ~3s
      - Warmup:             ~5s
      - 3 measure runs:     varies — average ~15s per request
        Single: ~5s × 3 = 15s
        Throughput: depends on parallel × time-per-request ≈ 20s
    """
    total = 0
    for spec in specs:
        startup_shutdown = 18
        warmup = 5
        if spec.profile == "single":
            measure = runs_per_combo * 6
        else:
            par = spec.params.get("parallel", 4)
            # Parallel requests run concurrently; one batch ~= max latency
            measure = runs_per_combo * (8 + par * 0.5)
        total += startup_shutdown + warmup + measure
    return int(total)
