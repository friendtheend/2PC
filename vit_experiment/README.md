# Reproducibility Guide (GELU Replacement: `poly` vs `d2poly`)

This repository is a SHAFT-based 2PC benchmark setup with a GELU replacement path for reproducible comparisons.

The goal of this README is to let others reproduce results after replacing GELU with `d2poly`.

## 1. Scope

This guide focuses on:
- GPT-2 private inference (`examples/text-generation`)
- ViT-B/16 private inference (`examples/image-classification`)
- GELU ablation between `poly` and `d2poly`

## 2. What Changed 

The GELU replacement is implemented in:
- `crypten/common/functions/approximations.py`

`d2poly` is wired as a GELU method and can be selected at runtime.

Current default config is in:
- `configs/default.yaml`


## 3. Environment

Use the original SHAFT environment (same dependency strategy as SHAFT).


## 4. Reproduce GELU Replacement Experiments

### 4.1 Activate environment


### 4.2 GPT-2 (`len_data=64`, `length=1`)

Run `poly`:

```bash
cd examples/text-generation
GELU_METHOD=poly python run_generation_private.py \
  --model_type=gpt2 \
  --model_name_or_path=openai-community/gpt2 \
  --len_data 64 \
  --length 1 \
  --estimate_mode total \
  --report_cost \
  2>&1 | tee /tmp/shaft_gpt2_l64_poly.log
```

Run `d2poly`:

```bash
cd examples/text-generation
GELU_METHOD=d2poly python run_generation_private.py \
  --model_type=gpt2 \
  --model_name_or_path=openai-community/gpt2 \
  --len_data 64 \
  --length 1 \
  --estimate_mode total \
  --report_cost \
  2>&1 | tee /tmp/shaft_gpt2_l64_d2poly.log
```

### 4.3 ViT-B/16 (`224x224`, single sample)

Run `poly`:

```bash
cd examples/image-classification
GELU_METHOD=poly REPORT_COST=1 EVAL_SAMPLES=1 \
  bash bench_vitb16_2pc_total.sh | tee /tmp/shaft_vit_poly.log
```

Run `d2poly`:

```bash
cd examples/image-classification
GELU_METHOD=d2poly REPORT_COST=1 EVAL_SAMPLES=1 \
  bash bench_vitb16_2pc_total.sh | tee /tmp/shaft_vit_d2poly.log
```

## 5. Metrics to Report

The logs already print component-level metrics. Commonly used fields:

- End-to-end latency
  - GPT-2: `running time: ...`
  - ViT: `BENCH_RESULT ... latency_sec=...`
- Linear latency
  - `embedding_time + matmul_time + conv_time`
- Activation latency
  - `softmax_time + gelu_time + layernorm_time + tanh_time`
- Total communication
  - `total_comm_bytes`
- Activation communication
  - `softmax_comm_bytes + gelu_comm_bytes + layernorm_comm_bytes + tanh_comm_bytes`
- Total communication rounds
  - `total_comm_rounds`
- Activation communication rounds
  - `softmax_comm_rounds + gelu_comm_rounds + layernorm_comm_rounds + tanh_comm_rounds`
- GELU-only metrics
  - `gelu_time`, `gelu_comm_bytes`, `gelu_comm_rounds`


