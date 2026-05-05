# BERT, LLaMA, and GPT-2 Experiment Bundle

This bundle collects the BERT, LLaMA, and GPT-2 experiment code, scripts, and
measurement outputs used for the current paper tables. Table V is the
BERT-base/QNLI SHAFT-vs-BriLLMFlow comparison; ViT code may still exist in the
vendored SHAFT tree, but it is not part of the main Table V pipeline.

## Can This Run After Moving?

Mostly yes, for the paths this bundle is intended to preserve:

- BERT/SHAFT and LLaMA/SHAFT experiment source trees are included.
- BriLLMFlow GPU online linear-operator source is included and can be rebuilt.
- LLaMA2 hybrid logs, CSVs, projection notes, and LaTeX table rows are included.
- BERT-base SHAFT vs BriLLMFlow Table V scripts are included.
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
  - SHAFT/CrypTen experiment tree. The directory name is historical; the Table
    V runner uses its BERT text-classification path under
    `examples/text-classification/`.
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

- `brillmflow_2pc/`
  - BriLLMFlow Matrix Beaver Triple sources used by the linear online
    measurements, including the original CUDA online implementation and BMT GPU
    prototype sources.

- `scripts/`
  - One-command experiment wrappers for BERT Table V, LLaMA2, and GPT2.
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
python3 -m pip install "numpy<2" evaluate omegaconf
```

Install a CUDA-enabled PyTorch build matching the machine's CUDA driver/toolkit.
The exact command depends on the target CUDA version; do not install CPU-only
PyTorch for the private inference runs.

Model/data access:

- BERT Table V runs use `andeskyl/bert-base-cased-qnli` and GLUE/QNLI by
  default.
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

The main paper-table commands are:

```bash
scripts/run_table5_bert.sh
scripts/run_table5_llama2.sh
scripts/run_table5_gpt2.sh
```

Each command creates one timestamped output directory under `results/` and
writes raw logs plus machine-readable summaries. The BERT, LLaMA2, and GPT2 scripts
automatically build the BriLLMFlow online CUDA binary if it is missing. Set
`CUDA_ARCH=sm_86` or another architecture if the target GPU is not A6000-class.

For BERT Table V specifically, the intended one-command entrypoint is:

```bash
scripts/run_table5_bert.sh
```

It is an alias for `scripts/run_bert_hybrid_table.sh` and writes all BERT Table
V artifacts in one directory. The LLaMA2 and GPT2 aliases behave the same way
for their corresponding tables.

Expected output layout:

```text
results/bert_hybrid_*/
  env.txt
  shaft_bert_qnli_l128_comp.log
  shaft_bert_qnli_l128_report.log
  brillm_online_bert_*.log
  bert_brillm_raw_online.csv
  bert_brillm_perop.csv
  bert_hybrid_summary.csv
  bert_hybrid_summary.json
  bert_table.tex
  table5.tex

results/llama2_experiment_*/
  env.txt
  commands.txt
  shaft_llama2_seq8_8layer_fourier.log
  shaft_llama2_seq8_8layer_d2poly.log
  shaft_summary.csv
  brillm_online_llama_*.log
  brillm_llama2_raw_online.csv
  llama2_brillm_perop.csv
  llama2_hybrid_summary.csv
  llama2_hybrid_summary.json
  llama2_table.tex
  table5.tex

results/gpt2_hybrid_*/
  env.txt
  shaft_gpt2_l128_comp.log
  shaft_gpt2_l128_report.log
  gpt2_brillm_raw_online.csv
  gpt2_brillm_perop.csv
  gpt2_hybrid_summary.csv
  gpt2_hybrid_summary.json
  gpt2_table.tex
  table5.tex
```

If you only want to build the BriLLMFlow online binary manually:

```bash
CUDA_ARCH=sm_86 scripts/build_brillm_online.sh
```

### BERT Table V

Run the full BERT-base QNLI Table V pipeline:

```bash
scripts/run_table5_bert.sh
```

This runs:

- SHAFT BERT-base QNLI, seq=128, batch size 1, vanilla Fourier GELU, with
  `--report_cost`.
- BriLLMFlow online GPU GEMM measurements for the six BERT-base linear
  operators: `W_Q/K/V`, `W_O`, `W_1`, `W_2`, `Q@K^T`, and `scores@V`.
- CSV and LaTeX generation for the paper-style Table V.

Useful overrides:

```bash
OUT=results/bert_hybrid_test GELU_METHOD=fourier scripts/run_table5_bert.sh
BRILLM_CUDA_VISIBLE_DEVICES=0 scripts/run_table5_bert.sh   # single-GPU two-rank run
CUDA_ARCH=sm_90 scripts/run_table5_bert.sh                 # build for Hopper
```

Single-GPU (H200) run:

```bash
CUDA_VISIBLE_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
scripts/run_table5_bert.sh
```

Single-machine shared-memory MPI (recommended for local 2-party runs; avoids forced TCP loopback):

```bash
CUDA_VISIBLE_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
BRILLM_MPI_MODE=shm \
scripts/run_table5_bert.sh
```

### LLaMA2 Experiments

Run the full LLaMA2 table pipeline:

```bash
scripts/run_table5_llama2.sh
```

The wrapper runs both Fourier and d2poly SHAFT LLaMA2 logs, runs the BriLLMFlow
online linear operator logs, projects the 8-layer run to 32 layers, and writes:

```text
results/llama2_experiment_*/shaft_summary.csv
results/llama2_experiment_*/brillm_llama2_raw_online.csv
results/llama2_experiment_*/llama2_hybrid_summary.csv
results/llama2_experiment_*/llama2_table.tex
results/llama2_experiment_*/table5.tex
```

Single-GPU (H200) run:

```bash
CUDA_VISIBLE_DEVICES=0 \
SHAFT_CUDA_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
scripts/run_table5_llama2.sh
```

Single-machine shared-memory MPI (recommended for local 2-party runs):

```bash
CUDA_VISIBLE_DEVICES=0 \
SHAFT_CUDA_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
BRILLM_MPI_MODE=shm \
scripts/run_table5_llama2.sh
```

### GPT2-base Experiments

Run the GPT2-base seq=128 SHAFT vs BriLLMFlow table pipeline:

```bash
scripts/run_table5_gpt2.sh
```

The GPT-2 runner writes raw logs, CSV summaries, `gpt2_table.tex`, and
`table5.tex` under a new `results/gpt2_hybrid_*` directory. It also keeps the SHAFT-standard
`--comp` command log:

```bash
python run_generation_private.py \
  --model_type=gpt2 \
  --model_name_or_path=openai-community/gpt2 \
  --len_data 128 \
  --comp \
  --length 1
```

Single-GPU (H200) run:

```bash
CUDA_VISIBLE_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
scripts/run_table5_gpt2.sh
```

Single-machine shared-memory MPI (recommended for local 2-party runs):

```bash
CUDA_VISIBLE_DEVICES=0 \
BRILLM_CUDA_VISIBLE_DEVICES=0 \
BRILLM_MPI_MODE=shm \
scripts/run_table5_gpt2.sh
```

MPI mode notes:

```text
BRILLM_MPI_MODE=tcp      # force tcp,self over lo (legacy behavior)
BRILLM_MPI_MODE=shm      # force shared-memory BTL (vader,self; OpenMPI 4.x)
BRILLM_MPI_MODE=default  # let OpenMPI auto-select transport
```

Threading note:

```text
The BriLLMFlow online CUDA binary does not use OpenMP for its main online path.
Setting OMP_NUM_THREADS may have little effect compared with MPI transport mode.
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

- BERT Table V wrapper: `scripts/run_table5_bert.sh`
- LLaMA2 table wrapper: `scripts/run_table5_llama2.sh`
- GPT2 table wrapper: `scripts/run_table5_gpt2.sh`
- BERT table generator: `scripts/make_bert_hybrid_table.py`
- LLaMA2 table generator: `scripts/make_llama2_hybrid_table.py`
- GPT2 table generator: `scripts/make_gpt2_hybrid_table.py`
- BriLLMFlow online CUDA source: `brillmflow_2pc/BMT/gpu_matrix_beaver_online.cu`
