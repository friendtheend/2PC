from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Llama on ARC-Easy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import load_state_dict
    from transformers.utils import (
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME,
    )

    from .eval import evaluate_arc_easy
    from .training import apply_quadratic_poly

    def _load_checkpoint_weights(model, checkpoint_path: str) -> None:
        ckpt = Path(checkpoint_path)
        if ckpt.is_file():
            state_dict = load_state_dict(str(ckpt))
            model.load_state_dict(state_dict, strict=False)
            return

        if not ckpt.is_dir():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        def _load_from_index(index_path: Path) -> bool:
            if not index_path.exists():
                return False
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map = index_data.get("weight_map", {})
            shard_files = sorted(set(weight_map.values()))
            for shard_file in shard_files:
                shard_path = ckpt / shard_file
                if not shard_path.exists():
                    raise FileNotFoundError(f"Missing shard file: {shard_path}")
                shard_state = load_state_dict(str(shard_path))
                model.load_state_dict(shard_state, strict=False)
            return True

        for index_name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
            if _load_from_index(ckpt / index_name):
                return

        for weights_name in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
            weights_path = ckpt / weights_name
            if weights_path.exists():
                state_dict = load_state_dict(str(weights_path))
                model.load_state_dict(state_dict, strict=False)
                return

        raise FileNotFoundError(f"No model weights found in checkpoint: {checkpoint_path}")

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    poly_cfg = cfg.get("poly", {})
    eval_cfg = cfg.get("evaluation", {})

    model_id = model_cfg["model_id"]
    tokenizer_id = model_cfg.get("tokenizer_id", model_id)
    local_files_only = model_cfg.get("local_files_only", False)
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dtype_name = model_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if poly_cfg.get("enabled", False):
        model = apply_quadratic_poly(model, replace_odd_layers=poly_cfg.get("replace_odd_layers", True))

    checkpoint_path = args.checkpoint or cfg.get("checkpoint_path")
    if checkpoint_path:
        _load_checkpoint_weights(model, checkpoint_path)

    result = evaluate_arc_easy(model, tokenizer, eval_cfg)
    print(result)


if __name__ == "__main__":
    main()
