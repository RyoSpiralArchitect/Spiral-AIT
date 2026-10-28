import cProfile
import io
import json
import pstats
import tracemalloc
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

from .aif_core import ActionSpace, ActiveInferenceAgent, AgentConfig, ExecutionObservation
from .np_compat import np
from .checkpoint import save_checkpoint
from .gwm_bridge import AITGWMBridge
from .multilingual import AVAILABLE_LANGUAGES
from .np_stub import NUMERIC_BACKEND as STUB_NUMERIC_BACKEND
from .onepass_ait import GateDiagnostics, OnePassAIT, StudentTrainingConfig


class _ScalarLogWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = (log_dir / "events.jsonl").open("a", encoding="utf-8")
        self._tag_steps: Dict[str, int] = {}

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        if global_step is None:
            last = self._tag_steps.get(tag, -1)
            global_step = last + 1
        else:
            global_step = int(global_step)
        self._tag_steps[tag] = global_step
        record = {
            "type": "scalar",
            "tag": tag,
            "value": float(scalar_value),
            "step": global_step,
            "timestamp": time.time(),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _summary_writer() -> _ScalarLogWriter:
    log_root = Path(__file__).resolve().parent / "logs"
    run_dir = log_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return _ScalarLogWriter(run_dir)


@contextmanager
def _profiled_run(enabled: bool) -> Generator[Optional[Dict[str, Path]], None, None]:
    if not enabled:
        yield None
        return

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    profile_path = logs_dir / f"run_profile_{stamp}.prof"
    report_path = logs_dir / f"run_profile_{stamp}.txt"
    memory_path = logs_dir / f"run_profile_{stamp}.memory.txt"
    profiler = cProfile.Profile()
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    profiler.enable()
    try:
        yield {"stats": profile_path, "report": report_path, "memory": memory_path}
    finally:
        profiler.disable()
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        profiler.dump_stats(profile_path)
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats_stream = io.StringIO()
        stats.stream = stats_stream
        stats.print_stats(50)
        cpu_report = stats_stream.getvalue()
        with report_path.open("w", encoding="utf-8") as fh:
            fh.write(cpu_report)
        summary_lines = [line for line in cpu_report.splitlines() if line.strip()]
        if summary_lines:
            print("Top CPU functions (cumulative time):")
            for line in summary_lines[:10]:
                print(f"  {line}")
        top_memory = end_snapshot.compare_to(start_snapshot, "lineno")[:25]
        with memory_path.open("w", encoding="utf-8") as fh:
            fh.write("Top memory consumers (by cumulative size):\n")
            for entry in top_memory:
                fh.write(
                    f"{entry.traceback.format()}\n    size={entry.size / 1024:.1f} KiB, count={entry.count}\n"
                )
        if top_memory:
            print("Top memory allocations (cumulative size):")
            for entry in top_memory[:5]:
                location = entry.traceback[0]
                print(
                    f"  {location.filename}:{location.lineno} — "
                    f"{entry.size / 1024:.1f} KiB over {entry.count} blocks"
                )



def boundary_f1(ait: OnePassAIT, text: str, gold_segments: List[str]) -> float:
    gold_cuts = set()
    idx = 0
    for tok in gold_segments:
        idx += len(tok)
        if idx < len(text):
            gold_cuts.add(idx)
    pred_segments = ait.segment_text(text)
    pred_cuts = set()
    idx = 0
    for tok in pred_segments:
        idx += len(tok)
        if idx < len(text):
            pred_cuts.add(idx)
    tp = len(gold_cuts & pred_cuts)
    fp = len(pred_cuts - gold_cuts)
    fn = len(gold_cuts - pred_cuts)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _run(writer: Optional[_ScalarLogWriter] = None) -> None:
    ait = OnePassAIT(latent_dim=24, seed=4242)
    owns_writer = writer is None
    if writer is None:
        writer = _summary_writer()
    try:
        writer.add_scalar(
            "backends/stub_accelerated",
            0.0 if STUB_NUMERIC_BACKEND == "python" else 1.0,
            global_step=0,
        )
        if STUB_NUMERIC_BACKEND != "python":
            print(f"NumPy stub acceleration backend: {STUB_NUMERIC_BACKEND}")
        train_summary = ait.train_student(
            cfg=StudentTrainingConfig(
                lr=0.05,
                epochs=84,
                batch_size=4,
                validation_split=0.2,
                patience=10,
                hidden_dim=32,
                emb_dim=20,
                window=3,
                phase_lr=0.6,
                encoder_lr=1e-3,
                cache_sequences=False,
            ),
            languages=AVAILABLE_LANGUAGES,
            include_reflective=True,
            shuffle=True,
            seed=4242,
        )
        print("Student boundary head summary:", json.dumps(train_summary, ensure_ascii=False, indent=2))
        backend = train_summary.get("backend") if isinstance(train_summary, dict) else None
        encoder_backend = train_summary.get("encoder_backend") if isinstance(train_summary, dict) else None
        available_devices = train_summary.get("available_devices", {}) if isinstance(train_summary, dict) else {}
        if backend:
            writer.add_scalar(
                "backends/boundary_jit",
                1.0 if str(backend).startswith(("julia", "compiled")) else 0.0,
                global_step=0,
            )
            print("Boundary backend:", backend)
        if available_devices:
            writer.add_scalar(
                "backends/has_gpu",
                1.0
                if any(
                    any(dev.lower().startswith(prefix) for prefix in ("cuda", "gpu", "metal"))
                    for devs in available_devices.values()
                    for dev in devs
                )
                else 0.0,
                global_step=0,
            )
            print("Backend device inventory:", available_devices)
        if encoder_backend:
            writer.add_scalar(
                "backends/encoder_external",
                1.0 if str(encoder_backend).startswith(("julia", "r")) else 0.0,
                global_step=0,
            )
            print("Encoder backend:", encoder_backend)
        encoder_devices = []
        if isinstance(train_summary, dict):
            encoder_devices = train_summary.get("encoder_devices", [])
        if encoder_devices:
            print("Encoder devices:", encoder_devices)

        train_texts: List[str] = []
        train_segments: List[List[str]] = []
        if isinstance(train_summary, dict):
            if "cache_sequences" in train_summary:
                writer.add_scalar(
                    "boundary/cache_sequences",
                    float(bool(train_summary["cache_sequences"])),
                    global_step=0,
                )
                print("Sequence cache enabled:", bool(train_summary["cache_sequences"]))
            if "cached_sequences" in train_summary:
                writer.add_scalar(
                    "boundary/cached_sequences",
                    float(train_summary["cached_sequences"]),
                    global_step=0,
                )
                print("Cached sequences used:", train_summary["cached_sequences"])
            train_texts = train_summary.get("dataset_texts", [])
            train_segments = train_summary.get("dataset_segments", [])
        else:
            train_texts = []
            train_segments = []
        language_tags = (
            train_summary.get("dataset_tags", []) if isinstance(train_summary, dict) else []
        )
        lang_hist = train_summary.get("dataset_languages", {}) if isinstance(train_summary, dict) else {}
        lang_stats = (
            train_summary.get("dataset_language_stats", {})
            if isinstance(train_summary, dict)
            else {}
        )
        if hasattr(lang_hist, "items"):
            for lang, count in lang_hist.items():
                writer.add_scalar(f"data/language/{lang}", count, global_step=0)
        print("Training language histogram:", lang_hist)
        if hasattr(lang_stats, "items"):
            for lang_idx, (lang, stats) in enumerate(lang_stats.items()):
                if not isinstance(stats, dict):
                    continue
                mean_chars = stats.get("mean_chars")
                mean_tokens = stats.get("mean_tokens")
                mean_cpt = stats.get("mean_chars_per_token")
                if mean_chars is not None:
                    writer.add_scalar(
                        f"data/mean_chars/{lang}", float(mean_chars), global_step=0
                    )
                if mean_tokens is not None:
                    writer.add_scalar(
                        f"data/mean_tokens/{lang}", float(mean_tokens), global_step=0
                    )
                if mean_cpt is not None:
                    writer.add_scalar(
                        f"data/mean_chars_per_token/{lang}",
                        float(mean_cpt),
                        global_step=0,
                    )
        print("Training language statistics:", lang_stats)

        history = train_summary.get("history", []) if isinstance(train_summary, dict) else []
        for step, metrics in enumerate(history, start=1):
            if "train_loss" in metrics:
                writer.add_scalar("boundary/train_loss", metrics["train_loss"], step)
            if "val_loss" in metrics:
                writer.add_scalar("boundary/val_loss", metrics["val_loss"], step)
            if "val_f1" in metrics:
                writer.add_scalar("boundary/val_f1", metrics["val_f1"], step)

        seg_scores = [boundary_f1(ait, text, seg) for text, seg in zip(train_texts, train_segments)]
        print("Segmentation F1 per sample:", seg_scores)
        for idx, score in enumerate(seg_scores):
            writer.add_scalar("boundary/seg_f1", score, idx)

        prompt = (
            "Bob re‑examined Carol's motives and updated his provisional evaluation. "
            "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。"
        )

        bench_iters = 12
        latencies_ms: List[float] = []
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        profile_raw_path = logs_dir / f"encode_profile_{stamp}.prof"
        profile_report_path = logs_dir / f"encode_profile_{stamp}.txt"
        memory_report_path = logs_dir / f"encode_memory_{stamp}.txt"

        profiler = cProfile.Profile()
        tracemalloc.start()
        profiler.enable()
        for idx in range(bench_iters):
            iter_start = time.perf_counter()
            ait.encode(prompt)
            latency_ms = (time.perf_counter() - iter_start) * 1000
            latencies_ms.append(latency_ms)
            writer.add_scalar("latency/encode_ms", latency_ms, idx)
        profiler.disable()
        snapshot = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        bench_time = sum(latencies_ms) / max(1, len(latencies_ms)) / 1000
        sorted_latencies = sorted(latencies_ms)
        p95_idx = min(len(sorted_latencies) - 1, int(0.95 * (len(sorted_latencies) - 1)))
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0
        writer.add_scalar("latency/encode_ms_avg", bench_time * 1000, bench_iters)
        writer.add_scalar("latency/encode_ms_p95", p95_latency, bench_iters)
        writer.add_scalar("latency/encode_peak_mem_mb", peak_mem / (1024 * 1024), bench_iters)
        print(f"Average encode latency: {bench_time*1000:.2f} ms (p95={p95_latency:.2f} ms)")

        profiler.dump_stats(profile_raw_path)
        profile_buffer = io.StringIO()
        pstats.Stats(profiler, stream=profile_buffer).strip_dirs().sort_stats("cumulative").print_stats(25)
        profile_path = Path(str(__file__).replace("run_demo.py", "encode_profile.txt"))
        with profile_path.open("w", encoding="utf-8") as profile_file:
            profile_file.write(profile_buffer.getvalue())

        top_stats = snapshot.statistics("lineno")[:10]
        with memory_report_path.open("w", encoding="utf-8") as memory_file:
            memory_file.write(f"Current tracked memory: {current_mem / (1024 * 1024):.2f} MiB\n")
            memory_file.write(f"Peak tracked memory: {peak_mem / (1024 * 1024):.2f} MiB\n")
            memory_file.write("Top allocations by line:\n")
            for stat in top_stats:
                memory_file.write(f"  {stat}\n")

        print(
            "Encode profile saved to",
            f"{profile_raw_path} (raw)",
            f"{profile_report_path} (top 25)",
            f"{memory_report_path} (memory)",
        )
        diag: GateDiagnostics = ait.gate_diagnostics()
        writer.add_scalar("gate/mask_energy", diag.mask_energy, global_step=0)
        print("Gate trace preview:", diag.gate_trace[:10])
        print("Attention strength per layer:", [round(v, 3) for v in diag.attention_strength])

        decode_result = ait.segment_text(prompt, return_metadata=True)
        decode_backend = decode_result.get("backend_used") if isinstance(decode_result, dict) else None
        if decode_backend:
            print("Boundary decode backend:", decode_backend)

        enc = ait.encode(prompt)
        if enc["phase_local"].size:
            phase_vals = enc["phase_local"].tolist() if hasattr(enc["phase_local"], "tolist") else enc["phase_local"]
            flat_phase = [float(val) for row in phase_vals for val in row]
            phase_energy = sum(flat_phase) / max(1, len(flat_phase))
        else:
            phase_energy = 0.0
        writer.add_scalar("phase/mean_energy", phase_energy, global_step=0)

        policies = ait.policies
        vecs = ait.policy_vecs
        action_space = ActionSpace(names=policies, vecs=vecs)

        bridge = AITGWMBridge(ait, enc, obs_sigma=0.6)
        goal_vec = ait.goal_vec
        cfg = AgentConfig(horizon=3, planner="cem", cem_K=6, cem_elite_frac=0.25, cem_iters=1, obs_sigma=0.6)

        # This legacy demo has no external sensor/environment. Keep its toy
        # execution explicit, and never admit synthetic targets as real replay.
        simulated_observation_id = ait.last_observation_id
        def simulated_environment(prior, action, context, forecast):
            nonlocal simulated_observation_id
            simulated_observation_id += 1
            return ExecutionObservation(simulated_observation_id, forecast.mu.copy(),
                                        (bridge.obs_sigma ** 2) * np.eye(ait.latent_dim),
                                        source="legacy_demo_simulated_environment", simulated=True)

        agent = ActiveInferenceAgent(
            dim=ait.latent_dim,
            action_space=action_space,
            goal_vec=goal_vec,
            step_fn=bridge.step_fn,
            ctx_provider=bridge.ctx_from_prefix,
            r3_provider=bridge.r3_from_prefix,
            cfg=cfg,
            observation_fn=simulated_environment,
            on_observation=bridge.observe_execution,
        )

        log = {"train_summary": train_summary, "steps": []}
        for step in range(1, 5):
            plan = agent.plan()
            a0 = plan.actions[0]
            upd = agent.act_and_update(a0)
            log["steps"].append(
                {
                    "step": step,
                    "chosen_action": a0,
                    "rollout_actions": plan.actions,
                    "efe_agg": plan.aggregate,
                    "upd": upd,
                    "R3_mix": float(ait.R3_mix),
                }
            )

        out_path = Path(str(__file__).replace("run_demo.py", "integrated_log.json"))
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **log,
                    "segmentation_f1": seg_scores,
                    "encode_latency_ms": bench_time * 1000,
                    "gate_trace": diag.gate_trace,
                    "attention_strength": diag.attention_strength,
                    "gate_mask_energy": diag.mask_energy,
                    "phase_local_mean": phase_energy,
                    "decode_backend": decode_backend,
                    "decode_backend_fallbacks": decode_result.get("backend_fallbacks", [])
                    if isinstance(decode_result, dict)
                    else [],
                    "language_tags": language_tags,
                    "language_histogram": lang_hist,
                    "language_statistics": lang_stats,
                    "boundary_backend": backend,
                    "encoder_backend": encoder_backend,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        save_checkpoint(str(__file__).replace("run_demo.py", "checkpoint.json"), ait.state_dict())
        print("Integrated demo finished. Actions picked:", [e["chosen_action"] for e in log["steps"]])
    finally:
        if owns_writer and hasattr(writer, "close"):
            writer.close()


def main(profile: bool = True) -> None:
    with _profiled_run(profile) as profile_artifacts:
        _run()
        if profile_artifacts is not None:
            print(
                "cProfile stats written to:",
                f" {profile_artifacts['stats']} (raw)",
                f" {profile_artifacts['report']} (top 50)",
            )
            memory_path = profile_artifacts.get("memory")
            if memory_path is not None:
                print(
                    "Memory profile written to:",
                    f" {memory_path} (top 25 cumulative)",
                )


if __name__ == "__main__":
    main()
