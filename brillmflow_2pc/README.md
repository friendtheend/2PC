# BriLLMFlow 2PC Sources

This directory contains the BriLLMFlow Matrix Beaver Triple sources used for the
online linear-operator measurements in the paper tables.

- `BMT/PCG_matrix_beaver.cpp`: CPU offline Matrix Beaver triple generator.
- `BMT/gpu_matrix_beaver_online.cu`: GPU online Matrix Beaver phase. This is the
  current conservative implementation that uses wide accumulation in the GEMM
  kernel and outputs 64-bit shares.
- `BMT/Makefile`: build rules from the measurement tree.
- `BMT_GPU/`: GPU offline/online prototype sources.

Experimental variants such as `gpu_matrix_beaver_online_ring64.cu` are not
included here to avoid confusing them with the reported method.

## Build Notes

The online benchmark can be built with:

```bash
../scripts/build_brillm_online.sh
```

The CPU offline generator depends on the original local OT/MPC support headers
(`comm/*`, `crypto/*`, `ot/*`). Those dependencies were not present as regular
source directories in the measurement tree, so this bundle preserves the source
file and Makefile but does not vendor missing third-party/local OT libraries.
For reproducing the online table rows, use dummy Matrix Beaver files generated
by `scripts/create_dummy_matrix_pcg.py`.
