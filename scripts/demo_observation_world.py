#!/usr/bin/env python3
"""Replay text while retaining joint boundary/observation-regime beliefs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_active_context import load_model
from scripts.benchmark_observation_world import MODES, fingerprints, load_parent_study, mode_values
from scripts.demo_active_ait import DEFAULT_TEXT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig, ActiveTextSession, StringTextSource
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import json_hash, sha256


def replay(study, text, *, mode="adaptive", char_cost=0.002):
    state, channel, parent, settings = load_parent_study(study)
    model = load_model(state, seed=5042)
    study_config = json.loads((ROOT / "configs/observation-world-pilot.json").read_text())
    config = ActiveContextConfig(**{**settings, **mode_values(mode, study_config), "char_cost": char_cost})
    before = json_hash({name: model.state_dict()[name] for name in ("student", "encoder", "phase", "dynamics")})
    code = {**fingerprints(), "scripts/demo_observation_world.py": sha256(Path(__file__).read_bytes())}
    result = ActiveTextSession(model, channel, config).run(StringTextSource(text))
    after = json_hash({name: model.state_dict()[name] for name in ("student", "encoder", "phase", "dynamics")})
    # Actual transitions enter replay even with weight fitting disabled. Verify
    # protected weights independently of that intended observation bookkeeping.
    for name in ("student", "encoder", "phase"):
        if model.state_dict()[name] != state[name]:
            raise RuntimeError("Replay altered the frozen segmenter")
    for name in ("W1", "b1", "W_mu", "b_mu", "W_sigma", "b_sigma"):
        if model.dynamics.export_state()[name] != state["dynamics"][name]:
            raise RuntimeError("Replay altered the frozen latent weights")
    if result["metrics"]["world_model_updates"] != 0 or not result["done"] or "".join(result["tokens"]) != text:
        raise RuntimeError("Replay violated completion, reconstruction or frozen-weight contract")
    if code != {**fingerprints(), "scripts/demo_observation_world.py": sha256(Path(__file__).read_bytes())}:
        raise RuntimeError("Implementation changed during replay")
    return {"kind": "provided_text_observation_hypothesis_replay", "mode": mode,
            "parent_study": parent, "code_sha256": code, "config": asdict(config),
            "model_state_before_sha256": before, "model_state_after_sha256": after,
            "input_sha256": sha256(text.encode()), **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / ".cache/active-context-pilot-v1")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--mode", choices=MODES, default="adaptive")
    parser.add_argument("--char-cost", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = replay(args.study, args.text, mode=args.mode, char_cost=args.char_cost)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print("Provided text replay; Bayesian observation hypotheses; no runtime gold labels")
        for event in receipt["trace"]:
            forecast = event.get("outcome_forecast")
            if forecast is not None:
                before, after = forecast["informative_probability_before"], forecast["informative_probability_after"]
                regime = f"; P(informative)={before:.4f}→{after:.4f}" if before is not None else ""
                print(f"{event['event']:03d} read {event['acquired_chars']} chars; "
                      f"outcome NLL={forecast['outcome_nll']:.4f}{regime}")
        print("Tokens:", json.dumps(receipt["tokens"], ensure_ascii=False))
        print("Metrics:", json.dumps(receipt["metrics"], ensure_ascii=False))
        return receipt
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
