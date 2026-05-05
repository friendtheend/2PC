from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_id: str = "NousResearch/Llama-2-7b-chat-hf"
    use_fast_tokenizer: bool = True


@dataclass
class DataConfig:
    dataset: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"
    split: str = "train"
    text_key: str = "text"
    max_length: int = 512


@dataclass
class TrainConfig:
    output_dir: str = "outputs/llama_minimal"
    learning_rate: float = 2e-5
    epochs: int = 1
    train_batch_size: int = 1
    grad_accum: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = True
    compile: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class PolyConfig:
    enabled: bool = True
    replace_odd_layers: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    poly: PolyConfig = field(default_factory=PolyConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> AppConfig:
    cfg = AppConfig()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _merge_dataclass(cfg, raw)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    return parser


def apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.train_batch_size is not None:
        cfg.train.train_batch_size = args.train_batch_size
    return cfg
