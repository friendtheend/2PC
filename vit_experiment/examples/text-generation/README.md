# Private Inference Benchmarks
This directory contains two different benchmark styles:

- GPT-2 private 1-token generation benchmarks.
- Llama2 private forward profiling benchmarks.

The Llama2 path currently measures one encrypted forward pass on a fixed-length input. It does not run a full autoregressive decode loop or emit a generated token.
## Preparation
Install dependencies:
```bash
pip install -r requirements.txt
```
## Running Experiments
### GPT-2 1-token generation
Computation cost of private GPT-2 inference for a length-64 input:
```bash
bash test_gpt2_64_comp.sh
```
Communication cost of private GPT-2 inference for a length-64 input:
```bash
bash test_gpt2_64_comm.sh
```
Computation cost of private GPT-2 inference for a length-128 input:
```bash
bash test_gpt2_128_comp.sh
```
Communication cost of private GPT-2 inference for a length-128 input:
```bash
bash test_gpt2_128_comm.sh
```

Unified 2PC benchmark (latency + communication) for cross-system comparison:
```bash
bash bench_gpt2_2pc_total.sh
```

### Llama2 forward profile
Unified 2PC forward-profile benchmark for Llama2-7B (latency + communication of one encrypted forward pass):
```bash
bash bench_llama2_2pc_total.sh
```
Notes:
`bench_llama2_2pc_total.sh` defaults to `FP16=1` to reduce GPU memory pressure.
Use `SHAFT_CUDA_DEVICES=0,1` (logical CUDA IDs) to map rank0/rank1 to different GPUs.
This path accepts `LEN_DATA` as the fixed input length for the forward pass. It is useful for memory-limited profiling and coarse extrapolation, but it is not a true `prompt -> generate 1 token` benchmark.

Optional SiLU approximation override (for activation ablations):
```bash
SILU_METHOD=d2poly bash bench_llama2_2pc_total.sh
```

Compatibility note:
`bench_llama2_2pc_total.sh` keeps the older command name, but in this bundle it
profiles one encrypted LLaMA2 forward pass rather than a full autoregressive
decode loop.
