# Llama3 Training and Evaluation (Submission)

This folder keeps only the Llama3 training/evaluation path used in experiments.

## Dependencies

Install from repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install accelerate peft
```

Required runtime packages:

- torch
- transformers
- datasets
- pyyaml
- accelerate
- peft

## Train (Llama3 Repro)

```bash
cd ~/PolyTransformer-Submission
python -m llama.train_llama3_repro --config llama/configs/llama3.yaml
```

Resume from checkpoint:

```bash
python -m llama.train_llama3_repro \
  --config llama/configs/llama3.yaml \
  --resume_from_checkpoint outputs/llama3_8b_poly_act/checkpoint-1500
```

## Evaluate (ARC-Easy, Log-likelihood)

Poly checkpoint:

```bash
python -m llama.evaluate \
  --config llama/configs/eval_llama3_arc_easy.yaml \
  --checkpoint outputs/llama3_8b_poly_act/checkpoint-1500
```

Baseline (no poly replacement):

```bash
python -m llama.evaluate --config llama/configs/eval_llama3_baseline_arc_easy.yaml
```
