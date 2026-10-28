#!/usr/bin/env python3
"""Replay authored text through the verified active-context study controller."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_active_context import load_model
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig, ActiveTextSession, StringTextSource
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import json_hash, sha256

DEFAULT_TEXT = (
    "When evidence changes, revise. The next observation can overturn an earlier boundary. "
    "Wait when context is useful, then commit what the evidence supports. "
    "文脈を確かめてから決める。急がず、届いた情報で判断を更新する。"
    "看到新证据，再更新判断。需要更多上下文时就先观察。"
)


def load_study(directory):
    directory = Path(directory)
    names = ("plan.json", "summary.json", "channel.json", "channel_fit.json",
             "warm_started_model.json", "warm_start.json", "development_selection.json")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError("Study is incomplete; run scripts/benchmark_active_context.py to completion first. Missing: " + ", ".join(missing))
    payloads = {name: (directory / name).read_bytes() for name in names}
    data = {name: json.loads(payload) for name, payload in payloads.items()}
    plan, summary = data["plan.json"], data["summary.json"]
    identity = plan["contract_sha256"]
    if json_hash(plan["contract"]) != identity:
        raise ValueError("Study plan contract hash mismatch")
    for name in ("summary.json", "channel_fit.json", "warm_start.json", "development_selection.json"):
        if data[name]["contract_sha256"] != identity:
            raise ValueError("Study artifact identity mismatch: " + name)
    required_code = {"scripts/benchmark_active_context.py", *[
        "spiralreality_AIT_onepass_aifcore_integrated/integrated/" + name
        for name in ("active_context.py", "acquisition.py", "onepass_ait.py")]}
    if not required_code <= plan["contract"]["code_sha256"].keys():
        raise ValueError("Study plan lacks the active-controller implementation fingerprints")
    for relative, digest in plan["contract"]["code_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha256((ROOT / path).read_bytes()) != digest:
            raise ValueError("Current implementation differs from the frozen study: " + relative)
    channel_state, fitted = data["channel.json"], data["channel_fit.json"]
    warm, selection = data["warm_start.json"], data["development_selection.json"]
    if json_hash(channel_state) != fitted["channel_sha256"] or channel_state.get("frozen") is not True:
        raise ValueError("Study channel integrity/frozen-state mismatch")
    if sha256(payloads["warm_started_model.json"]) != warm["checkpoint_sha256"]:
        raise ValueError("Study world checkpoint integrity mismatch")
    if warm["warm_transitions_sha256"] != fitted["warm_transitions_sha256"]:
        raise ValueError("Study warm-up evidence mismatch")
    if summary["selection"] != selection or summary["channel_fit"] != fitted or summary["world_warmup"] != warm:
        raise ValueError("Study summary does not bind the supplied artifacts")
    if selection["char_cost"] not in plan["contract"]["config"]["char_cost_candidates"]:
        raise ValueError("Selected acquisition cost is outside the frozen grid")
    channel = ContextAcquisitionModel.from_state(channel_state)
    model = load_model(data["warm_started_model.json"], plan["contract"]["config"]["model_seed"])
    protected = {key: model.state_dict()[key] for key in ("student", "encoder", "phase")}
    if json_hash(protected) != warm["protected_segmenter_sha256"]:
        raise ValueError("Loaded segmenter differs from the protected study model")
    config = {**plan["contract"]["config"]["active"], "char_cost": selection["char_cost"]}
    return model, channel, ActiveContextConfig(**config), {"contract_sha256": identity,
            "checkpoint_sha256": warm["checkpoint_sha256"], "channel_sha256": fitted["channel_sha256"],
            "selected_char_cost": selection["char_cost"], "protected_segmenter_sha256": warm["protected_segmenter_sha256"]}


def replay(study, text, *, freeze_world=False):
    model, channel, config, identity = load_study(study)
    if freeze_world:
        config = replace(config, learn_dynamics=False)
    result = ActiveTextSession(model, channel, config=config).run(StringTextSource(text))
    if not result["done"] or "".join(result["tokens"]) != text:
        raise RuntimeError("Replay did not complete and preserve the provided text")
    if freeze_world and result["metrics"]["world_model_updates"]:
        raise RuntimeError("Frozen-world replay updated world weights")
    protected = {key: model.state_dict()[key] for key in ("student", "encoder", "phase")}
    if json_hash(protected) != identity["protected_segmenter_sha256"] or json_hash(channel.export_state()) != identity["channel_sha256"]:
        raise RuntimeError("Replay changed the frozen segmenter or acquisition channel")
    return {"kind": "provided_text_replay", "world_adaptation_enabled": config.learn_dynamics, **identity, **result}


def describe(receipt):
    print("provided_text_replay — authored/provided text; no network or human interaction")
    print("Unsupervised world adaptation:", "enabled" if receipt["world_adaptation_enabled"] else "frozen")
    for event in receipt["trace"]:
        candidate = next((item for item in event["candidates"] if item["action"] == event["action"]
                          and item["requested_chars"] == event["requested_chars"]), None)
        expected = candidate["expected_information_gain"] if candidate else None
        realized = event.get("realized_entropy_reduction")
        fmt = lambda value: "n/a" if value is None else f"{value:+.5f}"
        acquisition = f" observed={event.get('acquired_chars', 0)} chars" if event["action"] == "Acquire" else f" emitted={event.get('token_chars', 0)} chars"
        print(f"{event['event']:03d} {event['action']}{event['requested_chars'] or ''} →{acquisition}"
              f"; expected_IG={fmt(expected)}; signed_ΔH={fmt(realized)}"
              f"; matched={event.get('matched_forecast', False)}; R3={event['phase_mixing']:.5f}"
              f"; world_step={event['world_model_trained_steps']}")
    metrics = receipt["metrics"]
    print("Tokens:", json.dumps(receipt["tokens"], ensure_ascii=False))
    print(f"Summary: acquisitions={metrics['acquire_calls']}, model_calls={metrics['model_calls']},"
          f" matched_forecasts={metrics['forecast_evaluations']}, online_world_weight_updates={metrics['world_model_updates']},"
          f" trace_truncated={metrics['trace_truncated']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / ".cache/active-context-pilot-v1")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-world", action="store_true", help="Disable online world-weight updates during replay")
    args = parser.parse_args(argv)
    try:
        receipt = replay(args.study, args.text, freeze_world=args.freeze_world)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        describe(receipt)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return receipt


if __name__ == "__main__":
    main()
