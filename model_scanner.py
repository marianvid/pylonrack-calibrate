"""model_scanner.py — Scan HF Cache for GGUF models (shared with llama slot)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Quant formats that llama.cpp build b4282+ no longer supports directly.
# These are the legacy ARM-interleaved Q4_0 variants; modern llama.cpp does
# online repacking from plain Q4_0 and rejects the pre-packed files. Skip
# them in scan() so the calibrator does not try to load broken models.
_OBSOLETE_QUANT_SUFFIXES = (
    "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8",
)


@dataclass(frozen=True)
class GGUFModel:
    display_name: str
    full_path:    str
    size_gb:      float

    def __str__(self) -> str:
        return self.display_name


def _display_name(repo_dir: Path, gguf_file: Path) -> str:
    repo = re.sub(r"^models--", "", repo_dir.name).replace("--", "/")
    stem = gguf_file.stem
    short_repo = repo.split("/")[-1].lower()
    if stem.lower().startswith(short_repo):
        stem = stem[len(short_repo):].lstrip("-_")
    return f"{repo} / {stem}" if stem else repo


def _is_obsolete_quant(name: str) -> bool:
    """Return True for legacy ARM-interleaved Q4_0 variants no longer
    supported by recent llama.cpp builds."""
    return any(suffix in name for suffix in _OBSOLETE_QUANT_SUFFIXES)


def scan(hf_cache_path: Path) -> list[GGUFModel]:
    if not hf_cache_path.exists():
        return []
    models: list[GGUFModel] = []
    seen_real_paths: set[str] = set()   # dedup symlink + blob pointing to same file

    for gguf_file in sorted(hf_cache_path.rglob("*.gguf")):
        # Skip projection models (mmproj-*.gguf used for vision adapters).
        if gguf_file.name.startswith("mmproj"):
            continue
        # Skip legacy Q4_0_X_X formats — they fail to load on llama.cpp b4282+.
        if _is_obsolete_quant(gguf_file.name):
            continue
        # Dedup: HF cache may contain both a snapshot symlink AND a direct
        # blob (or two snapshot paths) pointing to the same physical file.
        # Use the resolved real path as identity. First one wins — sorted()
        # above gives us deterministic preference (blobs/ comes before
        # snapshots/ alphabetically).
        real_path = str(gguf_file.resolve())
        if real_path in seen_real_paths:
            continue
        seen_real_paths.add(real_path)

        repo_dir = gguf_file.parent
        for ancestor in gguf_file.parents:
            if ancestor.name.startswith("models--"):
                repo_dir = ancestor
                break
        models.append(GGUFModel(
            display_name = _display_name(repo_dir, gguf_file),
            full_path    = str(gguf_file),
            size_gb      = round(gguf_file.stat().st_size / (1024 ** 3), 1),
        ))
    return models
