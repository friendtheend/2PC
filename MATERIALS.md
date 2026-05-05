# Materials

This repository does not include large datasets, pretrained model weights, or generated checkpoints.

## Public Inputs

- `google/vit-base-patch16-224`: downloaded by Hugging Face Transformers in the training and evaluation scripts.
- `imagenet-1k`: loaded through Hugging Face Datasets. Access requires accepting the ImageNet terms and authenticating with Hugging Face in the execution environment.

Set `data.hf_cache_dir` in the YAML config if the cluster uses a shared Hugging Face cache. Leave it as `null` to use the default local cache.

## Generated Outputs

- Distillation checkpoints are written under `outputs/<experiment>/`.
- Metric summaries are written as `outputs/<experiment>/metrics.yaml`.
- DirectFit sweep results are written by `python -m qura.directfit_sweep --config configs/directfit_only.yaml`.

These files are reproducible from the commands in `README.md` and are intentionally not versioned.

## Private Or Local Paths

The configs avoid hard-coded personal paths. If you run on a cluster, set local paths in a private copy of the YAML file, for example:

```bash
cp configs/coevolution_12_layers.yaml configs/local.yaml
python -m qura.train --config configs/local.yaml
```

Do not commit `configs/local.yaml` if it contains local usernames, cache paths, or machine-specific storage locations.

