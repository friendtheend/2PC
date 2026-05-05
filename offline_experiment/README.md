# Offline Experiment (15k Baseline)

This directory freezes the CPU offline PCG setup that reproduced the
high-throughput baseline (~15k+ C elements/s on `1024x64x1024`, depending on
machine and MPI/OpenMP stack).

## Included Files

- `PCG_matrix_beaver.cpp` (copied from validated baseline tree)
- `Makefile` (matching baseline tree)
- `build/pcg_matrix_beaver` (baseline binary copy for immediate use)

## Run Command (validated pattern)

```bash
cd offline_experiment
echo "$(hostname) slots=64" > hostfile

unset OMP_DYNAMIC OMP_WAIT_POLICY OMP_PLACES OMP_PROC_BIND
export OMP_NUM_THREADS=16

mpirun -np 2 --hostfile hostfile \
  --map-by ppr:2:node:PE=16 --bind-to core --report-bindings \
  --mca pml ob1 --mca btl vader,self \
  ./build/pcg_matrix_beaver \
  --M 1024 --K 64 --N 1024 --bits 64 --channels 20 --batch 256 --no-verify
```

## Notes

- Prefer `OMP_NUM_THREADS=16` for this workload.
- `OMP_NUM_THREADS=32` may run but can significantly reduce throughput.
- If you recompile, keep this directory isolated from other BMT variants.
