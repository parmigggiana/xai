import contextlib
import cProfile
import datetime as _dt
import os
from pathlib import Path
import pstats
import time
from typing import Iterator, Optional

import torch
import json
import csv


# Default output directory for profiling artifacts
DEFAULT_PROFILING_DIR = "./outputs/profiling"


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextlib.contextmanager
def cprofile_ctx(
    name: str, out_dir: str = DEFAULT_PROFILING_DIR
) -> Iterator[cProfile.Profile]:
    """Context manager to run cProfile and write .prof and .txt summary.

    Args:
        name: A short run name used to name the output files.
        out_dir: Directory where profiling artifacts will be written.
    """
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_base = _ensure_dir(out_dir) / f"{name}-{ts}"
    prof_path = out_base.with_suffix(".prof")
    txt_path = out_base.with_suffix(".txt")

    pr = cProfile.Profile()
    pr.enable()
    try:
        yield pr
    finally:
        pr.disable()
        pr.dump_stats(str(prof_path))
        # Also emit a human-readable summary
        with open(txt_path, "w", encoding="utf-8") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(60)


@contextlib.contextmanager
def torch_profiler_ctx(
    name: str,
    out_dir: str = DEFAULT_PROFILING_DIR,
    wait: int = 1,
    warmup: int = 1,
    active: int = 5,
    activities: Optional[list[str]] = None,
):
    """Context manager for torch.profiler with a schedule and trace export.

    Exports traces compatible with TensorBoard/Chrome under `out_dir/run-<name>-<ts>/`.

    Call `profiler.step()` once per iteration within the context.
    """
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _ensure_dir(out_dir) / f"run-{name}-{ts}"

    if activities is None:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
    else:
        # Map strings to activities
        acts = []
        for a in activities:
            if a.lower() == "cpu":
                acts.append(torch.profiler.ProfilerActivity.CPU)
            elif a.lower() == "cuda":
                if torch.cuda.is_available():
                    acts.append(torch.profiler.ProfilerActivity.CUDA)
            else:
                raise ValueError(f"Unknown profiler activity: {a}")

    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=1
    )

    prof = torch.profiler.profile(
        activities=acts,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    )

    prof.__enter__()
    try:
        yield prof
    finally:
        prof.__exit__(None, None, None)


@contextlib.contextmanager
def timer(name: str = "block") -> Iterator[float]:
    """Lightweight wall-time timer that yields elapsed seconds on exit.

    Example:
        with timer("step") as t:
            ...
        print(f"step took {t:.3f}s")
    """
    start = time.perf_counter()
    try:
        yield 0.0  # value is unused by most callers; they read elapsed post-context
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {name}: {elapsed:.3f}s")


def summarize_preprocess_logs(
    log_dir: str = DEFAULT_PROFILING_DIR, out_csv: str | None = None
) -> str:
    """Aggregate preprocessing JSONL logs (preprocess-*.jsonl) into a CSV.

    Each JSONL line is expected to be produced by ImageDataset._profiling_log.

    Args:
        log_dir: Directory containing preprocess-*.jsonl files.
        out_csv: Optional path to write the aggregated CSV. If None, defaults to
                 '<log_dir>/preprocess_summary.csv'.

    Returns:
        The path to the written CSV file.
    """
    log_path = Path(log_dir)
    files = list(log_path.glob("preprocess-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No preprocess-*.jsonl found under {log_dir}")

    rows = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        rows.append(rec)
                    except Exception:
                        continue
        except Exception:
            continue

    if not rows:
        raise RuntimeError("No valid JSONL records found to summarize")

    # Normalize keys and write CSV
    cols = [
        "ts",
        "pid",
        "index",
        "image_file",
        "seg_file",
        "slice_idx",
        "seg_slice_idx",
        "t_load_img_s",
        "t_load_seg_s",
        "t_xform_img_s",
        "t_xform_seg_s",
        "img_shape",
        "img_dtype",
        "seg_shape",
        "seg_dtype",
    ]
    out_csv = out_csv or str(log_path / "preprocess_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            r = {k: r.get(k, None) for k in cols}
            writer.writerow(r)
    return out_csv
