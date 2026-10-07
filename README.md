
# SpiralReality-AIT — One-Pass Active-Inference-Inspired Text Segmentation (NN + CRF)

[![License](https://img.shields.io/github/license/RyoSpiralArchitect/SpiralReality-AIT)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![Profile](https://img.shields.io/badge/stack-NumPy_first%20%7C%20CPU_friendly-brightgreen)

**Active-inference-inspired, streaming-friendly text segmentation & lightweight labeling.  
NumPy features, a small learned NN + CRF head, and bounded streaming windows.**

- **Streaming:** configurable lookahead, token-preserving commits, and optional confidence-based waiting.
- **Tiny & Fast:** NumPy-first, CPU-friendly; minimal deps and low memory.  
- **Interpretable:** phase/energy style signals and gate diagnostics you can actually inspect.  
- **CRF Head:** small NN features → CRF decoding for clean boundaries.  
- **AIF Policy Selection (optional):** compare candidate boundaries using one shared CRF belief (`use_aif=True`); the historical EFE proxy remains available explicitly.
- **Reproducible:** disjoint train/test evaluation, checkpoint round trips, installable wheels, and API/container demos.

The current model is a research segmenter, not a pretrained language model. The
character-category NN, context MLP, lexical residual, and CRF learn from
supervision. By default, transformer attention/FFN weights are random fixed
features and phase/gate updates are heuristic. An opt-in NumPy path now trains
the context transformer from the CRF loss. See [the earlier upgrade evidence](docs/upgrade-2026-10-05.md)
and [the English/Japanese/Chinese experiment](docs/trilingual-context-2026-10-08.md)
for measured results and limits. The [policy and learning follow-up](docs/policy-learning-2026-10-08.md)
repairs optimizer scaling and compares document budgets and initialization seeds.

## Quick start

> After cloning this repo:
```bash
# option A: local install (editable)
pip install -e .

# option B: Docker (build local image)
docker build -t spiralreality-ait:latest .
docker run --rm -p 8000:8000 spiralreality-ait:latest
```

**Minimal Python sample:**
```python
# pip install -e .   # make package importable
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    OnePassAIT,
    StudentTrainingConfig,
)

model = OnePassAIT()  # starts untrained
model.train_student(cfg=StudentTrainingConfig(epochs=8))

text = "Streaming input… chunk by chunk…"
tokens = model.segment_text(text, use_aif=True)  # shared-posterior policy selection (optional)
print(tokens)
```

For your own labels, pass `texts` and `segments` with
`''.join(segments[i]) == texts[i]`, including whitespace. Mismatched labels are
rejected before training. An explicit empty dataset is an error.

Enable the experimental learned character/bigram residual with
`StudentTrainingConfig(lexical_buckets=4096, lexical_lr=0.05)`. It augments the
category head with a fixed-size signed hash table, uses the Python CRF path,
and is included in checkpoints. The default remains disabled; evaluate it on
your own held-out data before adoption.

To learn the contextual transformer as well, explicitly select the NumPy
encoder and enable `train_context_encoder`. Pass your aligned `texts` and
`segments`, as above:

```python
model = OnePassAIT(latent_dim=16, encoder_layers=1, encoder_heads=2,
                   encoder_backend="numpy")
model.train_student(texts, segments, cfg=StudentTrainingConfig(
    train_context_encoder=True, encoder_lr=0.01,
    phase_lr=0.0, lexical_buckets=4096,
))
```

This differentiates the CRF loss through the context MLP, attention, feed-forward
layers, layer normalization, and learned gates. Gradients are accumulated by
batch and included in global clipping; early-stopping snapshots and checkpoints
contain the learned encoder weights. The supervised path replaces heuristic
encoder-gate updates. `phase_lr=0` also freezes phase updates for a controlled
comparison. Character input vectors and phase signals are not differentiated.
Native encoders without the backward interface reject this option explicitly.
In the completed small-budget PUD pilot, this option slightly reduced F1 in all
three languages. It remains experimental; the report records the negative result
alongside calibration and streaming measurements.
The subsequent [repair comparison](docs/policy-learning-2026-10-08.md) improves
the fixed-encoder baseline and finds additional Japanese/Chinese gains from
encoder learning at its largest training budget. Those results still use the
previously inspected PUD data and require independent-corpus confirmation.

NumPy training now averages the data gradient and adds the reported L2 penalty
**before** global clipping. Duplicating a batch's mean gradient therefore leaves
the optimizer step unchanged, including regularization. Training summaries use
`optimizer_contract="mean_nll_l2_then_global_clip_v2"` and report gradient norms
and clipping counts per epoch. Existing checkpoint predictions are unchanged;
retraining can differ. Native optimizer kernels are outside this correction.

```python
result = model.segment_text("Inspect each boundary.", include_confidence=True)
print(result["spans"])  # character offsets and confidence at each token end
print(result["boundary_probabilities"])  # one CRF marginal per interior boundary
```

These marginals describe model uncertainty, not calibrated correctness.
Confidence and AIF require the Python CRF weights; a model fitted only by an
optional native backend must be retrained on the Python path before using them.

When `use_aif=True`, `SegmentationAIFConfig(selection_mode="posterior_risk")` is
the default: every candidate is scored against the unchanged model's marginals.
Re-decoding collects no new evidence, so its epistemic term is zero. The selected
candidate cannot have higher estimated boundary risk than the zero-bias candidate
under that belief; this is not a guarantee of actual F1 improvement. Exact ties
prefer zero bias. Optional false-positive/false-negative costs and a positive
`marginal_temperature` control this score; fit calibration on separate data.
Use `selection_mode="legacy_efe"` for the old Gaussian EFE proxy, including its
policy-specific observation noise and preference targets.

Checkpoints record which backend owns the fitted weights. Loading a Python
checkpoint preserves installed native handles for later checkpoint swaps,
while inference stays on the saved owner's weights. Legacy files containing
both native weights and a Python-context flag, or multiple native heads, are
ambiguous and are rejected. Migrate those files only with verified training
provenance; the old context flag alone does not identify the trained backend.

`encode()` keeps a least-recently-used cache with at most 16 entries and
32 MiB of NumPy array payloads, including attention diagnostics. Configure
`OnePassAIT(encode_cache_max_entries=16, encode_cache_max_bytes=32 * 1024 * 1024)`
to change these limits; either limit set to zero disables retention. A result
larger than the byte budget is returned without caching. These limits cover
retained cache arrays, not peak inference memory, Python object overhead, or
the current result/diagnostics. Cache policy is local to the instance and is
not part of a model checkpoint.

**One-command demo (compose)**  
```bash
# if docker-compose.yml is provided
docker compose up --build
# API : http://localhost:8000/   |   Dashboard (if enabled): http://localhost:5173/
```

## Streaming (chunked inference)

For long/continuous streams, use the built-in chunked segmenter to keep a bounded window:

```python
stream = model.streaming_segmenter(
    max_window_chars=512,
    lookahead_chars=64,
    context_chars=128,
    use_aif=True,  # optional
    min_boundary_confidence=0.8,  # optional; default 0 disables confidence waiting
)

out = []
out += stream.feed("Streaming input… ")
out += stream.feed("chunk by chunk…")
out += stream.flush()
print(out)
```

Short feeds wait for the configured lookahead. Unfinished or uncertain tokens
remain pending until more context arrives or `flush()` ends the stream.
`hard_split=True` can force a cut at the maximum pending-window size even below
the confidence threshold. With `hard_split=False`, an uncommittable full window
raises `BufferError` when further input would overflow; the failed feed leaves
the wrapper buffers unchanged. All emitted pieces preserve the input text.
Chunked results can differ from full-text decoding because context is bounded.
`stream.forced_split_count` counts committed capacity-forced cuts, survives
`flush()`, and is cleared by `reset()`. Failed feeds/flushes roll back the count
with the buffers.

## Benchmarks (local run)

The [repair comparison](docs/policy-learning-2026-10-08.md) uses three training
budgets and three seeds, compares the old and corrected optimizers, and selects
encoder learning rates on development data before test evaluation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_policy_learning.py --prepare-only \
  --output .cache/my-policy-learning-study
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_policy_learning.py --workers 3 \
  --output .cache/my-policy-learning-study
```

It requires the pinned source cache (or an explicit `--download`) and the
reference Git commit named in its config. This is an exploratory follow-up on
the previously inspected PUD data; independent-corpus confirmation remains open.

The three-language pilot pins PUD r2.18 sources, preserves document/translation
groups across ten folds, and compares frozen versus supervised context features.
Raw data, licenses, checkpoints, and detailed fold receipts stay in the local
cache; the repository stores hashes, compact paired scores, and aggregate
results. Downloads occur only with the explicit `--download` flag.

```bash
python scripts/benchmark_trilingual.py --download --prepare-only \
  --output .cache/my-trilingual-run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_trilingual.py --output .cache/my-trilingual-run
```

The prepared plan locks code, data, configuration, and document folds. A changed
contract requires a new output directory. Completed matching folds can be
resumed. This is a custom, small-training-budget cross-validation experiment
on translated PUD text, not the official CoNLL test score or a production
quality claim. [Protocol, attribution, and results](docs/trilingual-context-2026-10-08.md).
To reproduce the **historical** published pilot exactly, use commit
`76ac1ebc8a3af0da42a7537baab0e021b612a0c9`; its source hashes precede the optimizer
and default policy changes above. A new run on current code is a new experiment.

Reproduce and write `reports/benchmark_report.{json,md}`:

```bash
python3 -c "from spiralreality_AIT_onepass_aifcore_integrated.integrated.benchmark import run_benchmark; run_benchmark(output_dir='reports', max_samples=None, seed=5042)"
```

The benchmark groups exact duplicates, balances sample caps across languages,
and scores only held-out text. Internal early stopping uses a separate portion
of the training pool. Reports include partition hashes, per-language F1, an
independent whitespace/punctuation baseline, perturbations, and inference
ablations. Context-off ablations are not separately retrained models.

The earlier 0.9140 example mixed training and evaluation text; it is not a
generalization result. The bundled corpus has only 33 synthetic examples.
Current measurements and matched comparisons are linked in
[the upgrade evidence](docs/upgrade-2026-10-05.md).

To isolate historical AIF computation reuse with the same model weights and
outputs, the following tool explicitly selects `legacy_efe` for both paths:

```bash
python scripts/benchmark_aif_reuse.py --output reports/aif_reuse.json
```

AIF shares one encoder pass across its policy candidates and final decoding;
the CRF still evaluates each distinct bias. Posterior-risk scoring also decodes
the candidates. Historical timing receipts apply to their pinned implementation;
neither the new score nor the explicit legacy EFE mode guarantees better F1.

## Architecture (Mermaid)

```txt
+-----------------------+      +----------------------------------+
|  Text stream / batch  | ---> |  Phase & Feature Extractors      |
|  (ASR / LLM output)   |      |  (tiny NN + heuristics)          |
+-----------------------+      +----------------------------------+
                                        |
                                        v
                              +--------------------------+
                              |  Gate / Energy Shaping   |
                              |  (one-pass, no backtrack)|
                              +--------------------------+
                                        |
                                        v
                              +--------------------------+
                              |  CRF Head (Viterbi)      |
                              +--------------------------+
                                        |
                                        v
                              +--------------------------+
                              |  Segments + Labels       |
                              +--------------------------+

                                    \
                                     \---> Live diagnostics (optional)
```

## Typical use-cases
- **Live captioning / subtitle segmentation** (ASR → segment → display)  
- **Chat/monitoring pipelines** where latency & memory matter  
- **Post-processing for LLM outputs** (boundary cleanup before downstream steps)  
- **Low-resource deployments** (containers on edge / CPU-only environments)

## Why AIT?
- **Streaming & stability:** bounded windows retain provisional text while emitted segments stay committed.
- **Interpretability:** phase/gate signals + CRF give controllable, explainable boundaries.  
- **Small-footprint:** NumPy-first; no heavy frameworks required for inference paths.

## Policy scoring during segmentation

`segment_text(..., use_aif=True)` compares candidate decodes using expected
boundary error under a shared CRF belief. It collects no new observation and
assigns no epistemic reward. The Gaussian belief/EFE components remain available
for other research loops, and the earlier segmentation proxy can be requested
with `SegmentationAIFConfig(selection_mode="legacy_efe")`.

## Positioning (vs spaCy / fastText / tiny BERT)
- **Not a full NLP pipeline**: SpiralReality-AIT is a boundary-focused segmenter with streaming constraints and interpretable signals, not a general-purpose POS/NER stack.
- **Edge/ops-first**: minimal deps, predictable latency, optional native acceleration, easy to embed as a microservice.
- **Inspectable + controllable**: gate traces, attention diagnostics, and explicit policy choice are first-class outputs (useful for production debugging).

⸻

## Highlights
- Hybrid boundary learner: learnable character-category embeddings → shallow tanh block → binary CRF with
  Viterbi decoding, plus an optional trainable character/bigram residual. The context MLP
  learns over transformer features, optionally training the NumPy transformer too;
  phase feedback remains heuristic and encoder-gate feedback is heuristic only in
  the fixed-transformer path. Optional bridges in
  `integrated/boundary_cpp.py` and `integrated/boundary_julia.py` now expose device discovery so
  bespoke C++/Julia/R implementations can advertise CUDA/Metal targets and accept `device_preference`
  hints while falling back to the pure NumPy student when unavailable.  The compiled C++ stub
  honours `SPIRAL_BOUNDARY_DEVICE` (falling back to `SPIRAL_DEVICE` or `SPIRAL_DEFAULT_DEVICE`) so
  deployments can steer default accelerator selections without touching the training code.
- Spectral transformer encoder: `integrated/encoder.py` upgrades the old toy adapter to a
  multi-head, layer-normalised NumPy transformer with phase-aware FiLM modulation.  It keeps track
  of gate-aware attention maps and reports its device inventory so external backends (or future GPU
  shims) can slot in without touching the One-Pass logic.  The optional
  `spiral_transformer_cpp` build now installs alongside the Python wrapper and surfaces
  compile-time CUDA/ROCm/MPS hints through `device_inventory()`/`set_device()` so GPU-capable
  targets can be selected without adding a PyTorch dependency while honouring
  `SPIRAL_TRANSFORMER_DEVICE`/`SPIRAL_DEVICE` overrides when choosing a default accelerator.
- Sequence caching: `StudentTrainingConfig(cache_sequences=False)` avoids keeping every
  derived training sequence in memory. Texts and labels are still materialized. Cached
  and uncached modes both refresh features that depend on changing phase parameters.
  Training summaries expose partition indices, restored best-epoch metrics, and processed tokens.
- Phase-aware encoding: curvature-derived local features are folded into the positional signal and
  boundary probabilities seed a gated attention mask.
  The encoder loader (`integrated/encoder_backends.py`) also probes for C++/Julia/R transformer
  adapters, mirroring the boundary student's backend selection so joint training can ride compiled
  kernels.
- Learned latent dynamics: a small MLP (see `integrated/dynamics.py`) distils the handcrafted
  transition rule and powers `OnePassAIT.predict_next` once sufficient experience has been
  collected.
- Deployment ready: a lightweight, dependency-free server (`server/main.py`) exposes `/health`
  and a WebSocket stream for boundary diagnostics, while `integrated/checkpoint.py` serialises
  model state to JSON for scripted usage.
- Instrumentation: `OnePassAIT.gate_diagnostics()` surfaces gate traces, attention energy, and gate
  mask strength.  `integrated/run_demo.py` streams structured JSON scalars for training loss/F1,
  latency, gate energy, and phase statistics while persisting checkpoints/logs for inspection.
- Visualization ready: `notebooks/run_demo.ipynb` mirrors the demo with Matplotlib/Seaborn hooks for
  attention maps, phase traces, and gate overlays.
- Curated corpora: the demo now assembles a multilingual dataset that augments the reflective
  English/Japanese anchors with curated Spanish, French, German, and Chinese narratives. The helper
  utilities in `integrated/multilingual.py` register the segments so the trainer and tests can reuse
  them consistently while exposing language histograms and per-language length/token statistics for
  rapid dataset audits.
- Licensed dataset export: `integrated/corpus.py` exposes `corpus_license()`/`corpus_catalog()` so the
  reflective and multilingual corpora can be redistributed under CC‑BY‑4.0 with per-language
  summaries for reporting or downstream tooling.
- Robustness benchmarking: `integrated/benchmark.py` trains the boundary student, applies dialect,
  noise, and tempo perturbations via `integrated/augmentation.py`, reports segmentation F1, p95
  latency, np_stub vs NumPy error, and writes JSON/Markdown summaries for dashboards.

## Layout
- `integrated/aif_core/` — compact Active Inference Core v2.
- `integrated/onepass_ait.py` — learnable phase basis, boundary NN+CRF, latent dynamics, diagnostics.
- `integrated/boundary.py` / `phase.py` / `encoder.py` / `dynamics.py` — modular components powering
  the student and latent model.
- `integrated/gwm_bridge.py` — binds One‑Pass AIT to AIF (ctx & step hooks).
- `integrated/run_demo.py` — end‑to‑end run; writes `integrated_log.json`, scalar logs, and a
  checkpoint.
- `notebooks/run_demo.ipynb` — interactive variant of the demo with visualization scaffolding.
- `tests/` — segmentation quality + encode latency regression tests.
- `.github/workflows/ci.yml` — GitHub Actions workflow (compile check + unit tests).
- `integrated/run_demo.py` — end‑to‑end run; writes `integrated_log.json` and a checkpoint.
- `tests/` — segmentation quality + encode latency regression tests.
- `.github/workflows/ci.yml` — GitHub Actions workflow (compile check + unit tests).
  
## Overview

This directory contains an integration of the "onepass" text-processing experiments with an aif_core component. The implementation is primarily NumPy-based and demonstrates a one-pass (online) processing pipeline that combines segmentation (boundary detection), phase-based local features, and a toy transformer-style encoder to produce contextualized embeddings.
tention

## Main files

- onepass_ait.py
  - Core implementation. Contains BoundaryStudent (boundary detector), phase feature computation, ToyTransformerAdapter (a minimal attention-style encoder), and OnePassAIT (integration logic).
- run_demo.py
  - Demo / example runner that shows how to run onepass_ait on sample text and visualize or print outputs.
- gwm_bridge.py
  - A bridging wrapper for connecting to external modules or alternative implementations.
- aif_core/
  - A directory for related core functionality if present; see that directory for more details.

## Minimal dependencies

- Python 3.9+
- numpy

Install example:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy
```

## Native acceleration builds

The project ships optional C++ and Julia accelerators that dramatically cut
latency for the numeric helpers and transformer adapter.  They are disabled by
default; build them explicitly when deploying on hosts with a native compiler or
a Julia runtime installed.

### C++ numeric/transformer modules

1. Install a C++17 toolchain and CMake (3.20+ recommended).
2. Activate your Python environment and install `pybind11`:

   ```bash
   pip install pybind11
   ```

3. Build the numeric helpers and transformer module in place:

   ```bash
   python native/cpp/setup_spiral_numeric_cpp.py build_ext --inplace
   python native/cpp/setup_spiral_transformer_cpp.py build_ext --inplace
   ```

   The build drops shared libraries next to the Python wrappers.  Set
   `SPIRAL_NUMERIC_BACKEND=cpp` to force the numeric stub to use the compiled
   implementation.  To prevent Python fallbacks when the compiled backend raises
   (for example, in latency-sensitive inference), set
   `SPIRAL_NUMERIC_STRICT=1`.

### Julia helper modules

1. Install Julia 1.9 or newer.
2. Instantiate the project dependencies:

   ```bash
   julia --project=native/julia -e 'using Pkg; Pkg.instantiate()'
   ```

3. (Optional) Precompile the modules to trim warm start latency:

   ```bash
   julia --project=native/julia -e 'using Pkg; Pkg.precompile()'
   ```

4. Set `JULIA_PROJECT=native/julia` before launching Python so `juliacall`
   resolves the modules under `native/julia/`.  The loader in
   `spiral_transformer_julia.py` automatically includes `SpiralTransformerJulia.jl`
   and caches the resulting module.

The demo trains the boundary student on the multilingual corpus by default. Use
`OnePassAIT.train_student(languages=("es", "ja"), include_reflective=False)` to target specific
languages programmatically.

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `logs/` → JSONL scalar logs describing training/evaluation traces.
## Quick start

1. Create and activate a virtual environment and install dependencies (see above).
2. Run the demo from the repository root:

```bash
python spiralreality_AIT_onepass_aifcore_integrated/integrated/run_demo.py
```

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `checkpoint.json` → JSON checkpoint for reloading through the diagnostics service.

## Native builds

The optional native helpers provide a faster execution path for the numeric
kernels. They are selected automatically when present; set
`SPIRAL_NUMERIC_BACKEND=cpp-strict` to require the compiled backend at runtime.

### C++ numeric helpers

1. Install build tooling (`cmake`, a C++17 compiler, and the Python headers).
2. From the repository root run:

   ```bash
   python native/cpp/setup_spiral_numeric_cpp.py build_ext --inplace
   ```

3. Verify the module loads:

   ```bash
   python -c "import spiral_numeric_cpp; print(spiral_numeric_cpp.__doc__)"
   ```

The build emits `spiral_numeric_cpp.*.so` alongside the integrated package.

### Julia numeric backend

1. Install Julia 1.9 or newer.
2. Instantiate the project environment:

   ```bash
   julia --project=native/julia -e 'using Pkg; Pkg.instantiate()'
   ```

3. Precompile the module once to shorten the first import:

   ```bash
   julia --project=native/julia -e 'using SpiralNumericJulia; SpiralNumericJulia.prewarm()'
   ```

Setting `SPIRAL_NUMERIC_BACKEND=julia` enables the Julia kernels when
available; `SPIRAL_NUMERIC_BACKEND=julia-strict` skips the Python fall-back.

## Packaging and Distribution

* Python packaging is configured via [`pyproject.toml`](./pyproject.toml). Use
  the helper scripts in [`packaging/`](./packaging) to build wheels locally or
  inside the manylinux Docker image.
* Build a wheel on the host:

  ```bash
  ./packaging/build_wheel.sh --version 0.1.0
  ```

* Produce manylinux wheels:

  ```bash
  ./packaging/build_manylinux_wheels.sh --version 0.1.0
  ```

## Docker Images

The [`docker/`](./docker) directory contains the runtime and manylinux builder
Dockerfiles. Build and push the runtime image with:

```bash
export IMAGE_TAG=ghcr.io/spiralreality/spiralreality-ait:latest
docker build -f docker/runtime.Dockerfile -t "$IMAGE_TAG" .
docker push "$IMAGE_TAG"
```

Refer to [`docker/README.md`](./docker/README.md) for more details.

Endpoints: `/health`, `/train`, `/segment`, `/encode`, `/load`.

## Tests & CI
```bash
pip install -e . pytest build
python -m pytest -q
python -m build
```

The demo trains the boundary student on the multilingual corpus by default. Use
`OnePassAIT.train_student(languages=("es", "ja"), include_reflective=False)` to target specific
languages programmatically.

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `logs/` → JSONL scalar logs describing training/evaluation traces.
- `checkpoint.json` → JSON checkpoint for reloading through the diagnostics service.

## Whitepaper Evaluation Pipeline

The repository ships with a reproducible workflow for building the latency/F1/robustness report in
`docs/whitepaper/`.

1. Generate raw metrics and CSV exports:

   ```bash
   python scripts/run_evaluation.py --output reports/evaluation
   ```

2. Produce SVG figures (pure-Python implementation, no external plotting stack required):

   ```bash
   python docs/whitepaper/generate_figures.py --input reports/evaluation/evaluation_metrics.json --output-dir reports/evaluation/figures
   ```

The historical whitepaper manuscript is a separate, static draft: neither the
commands above nor its PDF builder update the manuscript's numbers or figure
references. After reviewing and updating that manuscript explicitly, build it:

   ```bash
   # Requires matplotlib >= 3.7. Install via `pip install matplotlib`.
   python docs/whitepaper/build_whitepaper.py
   ```

To deliberately regenerate the historical whitepaper data and figures in place
(the manuscript text still requires manual review):

   ```bash
   make whitepaper
   ```

Fresh evaluation defaults to `reports/evaluation/`; `make whitepaper` explicitly
overwrites the derived data/figures under `docs/whitepaper/`. A release
checklist describing publication gating, DOI management, and GitHub Release hygiene is available at
`docs/whitepaper/release_checklist.md`.


## Native backends

Optional compiled backends live under `native/`.  The C++ variant builds a
`spiral_boundary_cpp` extension with pybind11; see `native/cpp/README.md` for
instructions.  A companion `spiral_boundary_gpu` module mirrors the API while
surfacing compiled accelerator targets (CUDA, ROCm/HIP, Metal) to Python so the
runtime can route training telemetry through native code even before GPU
kernels land.  The Julia implementation in `native/julia/SpiralBoundaryJulia.jl`
performs a similar device probe for `CUDA.jl`, `AMDGPU.jl`, and `Metal.jl` and
exposes the selected device in its summaries.  When any compiled module is on
the Python path the loader in `integrated/boundary_{cpp,julia}.py` will activate
it automatically and the NumPy trainer becomes a safety net rather than the
primary implementation.

## Real-time diagnostics stack

A lightweight, pure-Python diagnostics server and Vite dashboard are included
for streaming boundary inference:

```bash
docker compose up --build
```

- Backend service: `server/main.py` exposes `/health` and a `/ws` WebSocket that
  streams boundary segments plus `GateDiagnostics` metrics for arbitrary text
  without requiring external Python packages.
- Frontend app: `frontend/` connects to the WebSocket, renders gate traces and
  boundary probabilities, and can be served locally via `npm run dev`.
- Compose demo: `docker-compose.yml` wires both images for a one-command local
  experience. The dashboard becomes available at <http://localhost:5173> with
  the API at <http://localhost:8000>.

To run the diagnostics server without Docker execute:

```bash
python -m server.main
```

For a notebook walkthrough see [`notebooks/TUTORIAL.md`](notebooks/TUTORIAL.md)
and [`notebooks/websocket_demo.ipynb`](notebooks/websocket_demo.ipynb).
