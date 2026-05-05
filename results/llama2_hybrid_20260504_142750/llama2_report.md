# LLaMA2 Hybrid Measurement Summary

Output dir: `/home/ylu18/luoresearch/survay/PolyTransformer-Submission/results/llama2_hybrid_20260504_142750`

| Op | (M,K,N) | count | Time(ms) | Comm(MB) | BW(Gbps) |
|---|---:|---:|---:|---:|---:|
| W_Q/K/V | (8,4096,4096) | 96 | 11226.3 | 12910.1 | 9.20 |
| W_O | (8,4096,4096) | 32 | 3742.2 | 4303.4 | 9.20 |
| W_gate/up | (8,4096,11008) | 64 | 19532.2 | 23102.2 | 9.46 |
| W_down | (8,11008,4096) | 32 | 9858.8 | 11565.3 | 9.38 |
| Q@K^T | (8,128,8) | 1024 | 88.3 | 16.8 | 1.52 |
| scores@V | (8,8,128) | 1024 | 72.3 | 8.9 | 0.99 |

| Metric | SHAFT Fourier | BriLLMFlow hybrid | Speedup/reduction |
|---|---:|---:|---:|
| Linear time | 149897.3 ms | 44520.2 ms | 3.37x |
| Linear comm | 104498.0 MB | 51906.6 MB | 2.01x less |
| Non-linear time | 21711.7 ms | 19524.7 ms | 1.11x |
| Non-linear comm | 4743.2 MB | 2307.7 MB | 2.06x less |
| of which SiLU time | 2770.8 ms | 157.8 ms | 17.56x |
| of which SiLU comm | 2477.5 MB | 42.0 MB | 59.00x less |
| Online E2E | 171609.0 ms | 64044.9 ms | 2.68x |
| Online comm | 109241.3 MB | 54214.3 MB | 2.01x less |
