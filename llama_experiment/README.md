# Llama2 2PC Profiling

This README documents only the Llama2 private-inference profiling path used in this project.

## Scope

- Entry point: `examples/text-generation/run_generation_private.py`
- Model: `meta-llama/Llama-2-7b-hf`
- Mode: private forward profiling (not full autoregressive decode loops)
- Typical settings: `WORLD_SIZE=2`, `FP16=1`, `len_data=8`, `max_layers=16`
- Methods compared:
  - QURA: `d2poly`
  - Original baseline: `fourier`

## Environment

The base software stack follows SHAFT's setup: Linux (Ubuntu 22.04), Python 3.10, CUDA, and a CUDA-enabled PyTorch and so on.

We additionally pin `numpy<2` to avoid converter/runtime incompatibilities in the ONNX/CrypTen path.

You also need Hugging Face access to `meta-llama/Llama-2-7b-hf`.
For 2PC runs, we set `WORLD_SIZE=2`; in our local runs both ranks are mapped to one visible GPU slot (`CUDA_VISIBLE_DEVICES=0`, `SHAFT_CUDA_DEVICES=0`).

## Single Run (16-layer)

Run from `examples/text-generation`:

```bash
WORLD_SIZE=2 \
CUDA_VISIBLE_DEVICES=0 \
SHAFT_CUDA_DEVICES=0 \
PYTORCH_ALLOC_CONF=max_split_size_mb:64 \
FP16=1 \
SILU_METHOD=d2poly \
CRYPTEN_ONNX_FORCE_DISK=1 \
CRYPTEN_KEEP_PYTORCH_MODEL=0 \
python run_generation_private.py \
  --model_type llama \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --len_data 8 --length 1 \
  --estimate_mode total --fp16 --report_cost \
  --max_layers 16 \
  --silu_method d2poly
```

To run the original baseline, change both `SILU_METHOD` and `--silu_method` to `fourier`.

## Metrics in Logs

- `total_running_time`, `silu_time`
- `total_comm_bytes`, `silu_comm_bytes`
- `total_comm_rounds`, `silu_comm_rounds`
