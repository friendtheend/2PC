# ViT, LLaMA, and GPT-2 Experiment Bundle

This bundle collects the ViT, LLaMA, and GPT-2 experiment code, scripts, and
measurement outputs used for the current paper tables.

## Can This Run After Moving?

Mostly yes, for the paths this bundle is intended to preserve:

- ViT/SHAFT and LLaMA/SHAFT experiment source trees are included.
- BriLLMFlow GPU online linear-operator source is included and can be rebuilt.
- LLaMA2 hybrid logs, CSVs, projection notes, and LaTeX table rows are included.
- ViT SHAFT image-classification scripts are included.
- GPT2-base SHAFT vs BriLLMFlow table scripts are included.
- Helper scripts are included for rebuilding and rerunning BriLLMFlow LLaMA2
  online linear operators with generated dummy PCG files.

It is not a fully self-contained machine image. A new machine still needs the
system and Python environment below, plus model access/cache. The CPU offline
PCG generator source is preserved, but its original local OT/MPC support
headers/libraries were not available as normal source directories in this
workspace, so the CPU offline generator is not guaranteed to rebuild from this
bundle alone.

## Contents

- `vit_experiment/`
  - ViT/SHAFT-style experiment tree, including image-classification scripts.
  - Large caches, compiled Python files, model checkpoints, and model binaries
    are excluded.

- `llama_experiment/`
  - LLaMA2-compatible SHAFT/CrypTen experiment path.
  - Includes the patched `run_generation_private.py`, CrypTen modules, configs,
    and text-generation/image-classification experiment scripts.

- `llama_training_scripts/`
  - LLaMA training/evaluation helper scripts and configs from
    `PolyTransformer-Submission/llama`.

- `results/llama2_hybrid_20260504_142750/`
  - LLaMA2 8-layer measured / 32-layer projected hybrid E2E results.
  - Includes raw SHAFT logs, BriLLMFlow online logs, CSV summaries, JSON summary,
    projection notes, and LaTeX table fragments.
  - Dummy PCG `.bin` files are excluded because they are generated artifacts.

- `HE_related/`
  - Minimal ViT/SPU-related files copied from `HE_experiment`, without vendoring
    the full SPU source tree.

- `brillmflow_2pc/`
  - BriLLMFlow Matrix Beaver Triple sources used by the linear online
    measurements, including the original CUDA online implementation and BMT GPU
    prototype sources.

- `scripts/`
  - One-command experiment wrappers for ViT, LLaMA2, and GPT2.
  - Helper scripts to build the BriLLMFlow online binary, generate dummy PCG
    files, parse SHAFT logs, parse BriLLMFlow logs, and generate table CSV/TeX.

- `MANIFEST.txt`
  - Full file list for reviewing what will be committed.

## Environment Requirements

Recommended hardware:

- Linux x86_64 machine.
- NVIDIA GPU(s). The LLaMA2 8-layer run was designed for two A6000-class GPUs,
  but the BriLLMFlow online linear microbenchmarks can run on one or two CUDA
  devices depending on memory and MPI rank mapping.
- Enough disk for generated dummy PCG files and logs. The scripts write under
  `results/` by default.

System packages:

- CUDA toolkit with `nvcc`.
- NVIDIA driver compatible with the CUDA toolkit.
- OpenMPI with `mpirun` and `mpicxx`.
- Python 3.10 is the safest target for the SHAFT/CrypTen code path.
- A C++17-capable host compiler accepted by `nvcc`.

Python packages:

```bash
python3 -m pip install -r requirements.polytransformer.txt
python3 -m pip install "numpy<2"
```

Install a CUDA-enabled PyTorch build matching the machine's CUDA driver/toolkit.
The exact command depends on the target CUDA version; do not install CPU-only
PyTorch for the private inference runs.

Model/data access:

- ViT runs use `google/vit-base-patch16-224` by default.
- LLaMA2 runs use `meta-llama/Llama-2-7b-hf`; the target machine needs accepted
  HuggingFace access and either a valid login token or a pre-populated HF cache.
- Model weights and HF cache files are intentionally not included in this
  bundle.

Network:

- The included LLaMA2 table logs record a local/noqueue network state, not a
  tc-enforced 200 Gbps rate limit.
- If you want a rate-limited rerun, configure it on the target machine before
  launching experiments, for example:

```bash
sudo tc qdisc replace dev lo root netem rate 200gbit limit 10000
tc qdisc show dev lo
```

Use the interface appropriate for the actual two-party deployment if not using
single-node loopback.

## Quick Start On A New Machine

From the bundle root:

```bash
scripts/check_env.sh
```

The three top-level experiment commands are:

```bash
scripts/run_vit_experiment.sh
scripts/run_llama2_experiment.sh
scripts/run_gpt2_hybrid_table.sh
```

Each command creates one timestamped output directory under `results/` and
writes raw logs plus machine-readable summaries. The LLaMA2 and GPT2 scripts
automatically build the BriLLMFlow online CUDA binary if it is missing. Set
`CUDA_ARCH=sm_86` or another architecture if the target GPU is not A6000-class.

Expected output layout:

```text
results/vit_experiment_*/
  env.txt
  commands.txt
  vit_d2poly_report.log
  vit_fourier_report.log
  vit_comp.log
  vit_comm.log
  shaft_summary.csv

results/llama2_experiment_*/
  env.txt
  commands.txt
  shaft_llama2_seq8_8layer_fourier.log
  shaft_llama2_seq8_8layer_d2poly.log
  shaft_summary.csv
  brillm_online_llama_*.log
  brillm_llama2_raw_online.csv

results/gpt2_hybrid_*/
  env.txt
  shaft_gpt2_l128_comp.log
  shaft_gpt2_l128_report.log
  gpt2_brillm_raw_online.csv
  gpt2_brillm_perop.csv
  gpt2_hybrid_summary.csv
  gpt2_table.tex
```

If you only want to build the BriLLMFlow online binary manually:

```bash
CUDA_ARCH=sm_86 scripts/build_brillm_online.sh
```

### ViT Experiments

The ViT path is under:

```text
vit_experiment/examples/image-classification/
```

Main entrypoints:

```text
run_image_classification_private.py
bench_vitb16_2pc_total.sh
test_vit_base_224_comp.sh
test_vit_base_224_comm.sh
```

Run the SHAFT ViT-B/16 private inference total-profile path:

```bash
cd vit_experiment/examples/image-classification
GELU_METHOD=d2poly REPORT_COST=1 EVAL_SAMPLES=1 \
  bash bench_vitb16_2pc_total.sh
```

Run the vanilla/Fourier-style baseline by changing `GELU_METHOD`:

```bash
cd vit_experiment/examples/image-classification
GELU_METHOD=fourier REPORT_COST=1 EVAL_SAMPLES=1 \
  bash bench_vitb16_2pc_total.sh
```

Run SHAFT's original separated comp/comm scripts:

```bash
cd vit_experiment/examples/image-classification
bash test_vit_base_224_comp.sh
bash test_vit_base_224_comm.sh
```

ViT uses `google/vit-base-patch16-224` by default in the bundled benchmark
script. The model weights are not included; the target machine needs HF access
or a populated cache. Logs are printed to stdout unless you redirect them, e.g.:

```bash
mkdir -p ../../results/vit_rerun
GELU_METHOD=d2poly REPORT_COST=1 EVAL_SAMPLES=1 \
  bash bench_vitb16_2pc_total.sh 2>&1 | tee ../../results/vit_rerun/vit_d2poly.log
```

### LLaMA2 Experiments

Rerun the LLaMA2 forward-shape BriLLMFlow online linear operators:

```bash
OUT=results/rerun_llama2_online_$(date +%Y%m%d_%H%M%S) \
  scripts/run_llama2_online_ops.sh
```

Run LLaMA2 SHAFT 2PC profiling from the LLaMA experiment tree:

```bash
cd llama_experiment/examples/text-generation
WORLD_SIZE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
SHAFT_CUDA_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=max_split_size_mb:64 \
FP16=1 \
SILU_METHOD=fourier \
CRYPTEN_ONNX_FORCE_DISK=1 \
CRYPTEN_KEEP_PYTORCH_MODEL=0 \
MAX_LAYERS=8 \
LEN_DATA=8 \
bash bench_llama2_2pc_total.sh
```

For the d2poly/MixPoly-style SiLU path, set `SILU_METHOD=d2poly`.

The one-command wrapper above runs both Fourier and d2poly SHAFT LLaMA2 logs
and the BriLLMFlow online linear operator logs, then writes:

```text
results/llama2_experiment_*/shaft_summary.csv
results/llama2_experiment_*/brillm_llama2_raw_online.csv
```

### GPT2-base Experiments

Run the GPT2-base seq=128 SHAFT vs BriLLMFlow table pipeline:

```bash
scripts/run_gpt2_hybrid_table.sh
```

The GPT-2 runner writes raw logs, CSV summaries, and `gpt2_table.tex` under a
new `results/gpt2_hybrid_*` directory. It also keeps the SHAFT-standard
`--comp` command log:

```bash
python run_generation_private.py \
  --model_type=gpt2 \
  --model_name_or_path=openai-community/gpt2 \
  --len_data 128 \
  --comp \
  --length 1
```

## Important Measurement Caveats

- LLaMA2 SHAFT results use the LLaMA-compatible adapter in `llama_experiment`;
  this is not untouched upstream SHAFT.
- LLaMA2 full model results are projected from an 8-block two-party run to
  32 blocks due to GPU memory limits.
- The LLaMA2 run uses 8 input tokens and one generated token.
- The recorded network state for the LLaMA2 bundle run is local/noqueue, not a
  tc-enforced 200 Gbps rate limit.
- HuggingFace model weights and local cache files are not included.
- External system packages are still required on a new machine: CUDA/NVCC,
  OpenMPI, Python/Conda dependencies for SHAFT/CrypTen, and HuggingFace model
  access/cache for LLaMA2.
- The CPU offline PCG generator source is included, but its original local
  OT/MPC support headers/libraries were not present as normal source
  directories in this workspace. The included online rerun path uses generated
  dummy PCG files for timing the online linear operators.

## Key Files

- ViT SHAFT script:
  `vit_experiment/examples/image-classification/bench_vitb16_2pc_total.sh`
- ViT SHAFT entrypoint:
  `vit_experiment/examples/image-classification/run_image_classification_private.py`
- LLaMA2 table:
  `results/llama2_hybrid_20260504_142750/llama2_table_bert_style.tex`
- LLaMA2 aggregate CSV:
  `results/llama2_hybrid_20260504_142750/llama2_hybrid_aggregate.csv`
- LLaMA2 per-op CSV:
  `results/llama2_hybrid_20260504_142750/brillm_llama2_forward_online_e2e_counts.csv`
- LLaMA2 projection notes:
  `results/llama2_hybrid_20260504_142750/projection_notes.txt`
- BriLLMFlow online CUDA source:
  `brillmflow_2pc/BMT/gpu_matrix_beaver_online.cu`
