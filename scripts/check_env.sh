#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== System tools =="
for tool in python3 nvcc mpirun mpicxx nvidia-smi; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf "%-12s %s\n" "$tool" "$(command -v "$tool")"
  else
    printf "%-12s MISSING\n" "$tool"
  fi
done

echo
echo "== Versions =="
python3 --version || true
nvcc --version | sed -n '1,4p' || true
mpirun --version | sed -n '1,2p' || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo
echo "== Python imports =="
PYTHONPATH="$ROOT/llama_experiment:$ROOT/vit_experiment:${PYTHONPATH:-}" python3 - <<'PY'
mods = ["torch", "torchvision", "transformers", "datasets", "evaluate", "numpy", "pandas", "scipy", "tqdm", "crypten"]
for name in mods:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name:12s} OK {ver}")
    except Exception as exc:
        print(f"{name:12s} MISSING/ERROR {exc}")
try:
    import torch
    print(f"cuda_available {torch.cuda.is_available()}")
    print(f"cuda_device_count {torch.cuda.device_count()}")
except Exception:
    pass
PY

echo
echo "== Local source checks =="
test -f "$ROOT/brillmflow_2pc/BMT/gpu_matrix_beaver_online.cu" && echo "BriLLMFlow online CUDA source OK"
test -f "$ROOT/llama_experiment/examples/text-generation/run_generation_private.py" && echo "LLaMA SHAFT entrypoint OK"
test -f "$ROOT/vit_experiment/examples/text-classification/run_glue_private.py" && echo "BERT SHAFT entrypoint OK"

echo
echo "Environment check finished. Missing tools/imports must be fixed before rerunning experiments."
