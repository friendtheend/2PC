from __future__ import annotations

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class QuadraticActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1e-3]))
        self.b = torch.nn.Parameter(torch.ones(1))
        self.c = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        a = self.a.to(x.device).to(dtype)
        b = self.b.to(x.device).to(dtype)
        c = self.c.to(x.device).to(dtype)
        return a * x.pow(2) + b * x + c


def _resolve_layers(model: torch.nn.Module):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
    ]
    for getter in candidates:
        try:
            return getter(model)
        except AttributeError:
            continue
    raise AttributeError("Cannot resolve transformer layers for this model")


def apply_quadratic_poly(model: torch.nn.Module, replace_odd_layers: bool = True) -> torch.nn.Module:
    layers = _resolve_layers(model)
    for idx in range(len(layers)):
        if (idx % 2 != 0) == replace_odd_layers:
            layers[idx].mlp.act_fn = QuadraticActivation()
    return model


def train(cfg) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id,
        use_fast=cfg.model.use_fast_tokenizer,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.bfloat16 if cfg.train.bf16 else None,
        trust_remote_code=True,
    )

    if cfg.poly.enabled:
        model = apply_quadratic_poly(model, replace_odd_layers=cfg.poly.replace_odd_layers)

    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    raw_ds = load_dataset(cfg.data.dataset, cfg.data.subset, split=cfg.data.split)

    def preprocess(batch):
        texts = [t for t in batch[cfg.data.text_key] if isinstance(t, str) and t.strip()]
        return tokenizer(texts, truncation=True, max_length=cfg.data.max_length)

    tokenized = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds.column_names)

    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.train_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=1,
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_pin_memory=cfg.train.dataloader_pin_memory,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)

    return {
        "train_loss": float(train_result.training_loss),
        "model": cfg.model.model_id,
        "poly_enabled": float(cfg.poly.enabled),
        "epochs": float(cfg.train.epochs),
        "batch_size": float(cfg.train.train_batch_size),
        "lr": float(cfg.train.learning_rate),
    }
