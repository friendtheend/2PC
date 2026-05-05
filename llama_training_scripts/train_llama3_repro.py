from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_state_dict
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
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
        lambda m: m.transformer.layers,
    ]
    for getter in candidates:
        try:
            return getter(model)
        except AttributeError:
            continue
    raise AttributeError("Cannot resolve transformer layers")


def apply_quadratic_poly(model: torch.nn.Module, replace_odd_layers: bool = True) -> torch.nn.Module:
    layers = _resolve_layers(model)
    for idx in range(len(layers)):
        if (idx % 2 != 0) == replace_odd_layers:
            layers[idx].mlp.act_fn = QuadraticActivation()
    return model


def compute_logic_distill_loss(student_out, teacher_out, attention_mask, input_ids, weights: dict) -> tuple[torch.Tensor, dict]:
    w_kl = float(weights.get("kl", 1.0))
    w_ce = float(weights.get("ce", 1.0))
    w_mse = float(weights.get("mse", 1.0))
    w_cos = float(weights.get("cosine", 1.0))

    mask = attention_mask.float()
    active = mask.sum().clamp_min(1.0)

    s_logits = student_out.logits.float()
    t_logits = teacher_out.logits.float()

    kl = F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    kl = (kl * mask).sum() / active

    shift_logits = s_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

    mse = torch.tensor(0.0, device=s_logits.device)
    cos = torch.tensor(0.0, device=s_logits.device)
    if student_out.hidden_states is not None and teacher_out.hidden_states is not None:
        num_layers = min(len(student_out.hidden_states), len(teacher_out.hidden_states))
        for i in range(1, num_layers):
            s_h = student_out.hidden_states[i].float()
            t_h = teacher_out.hidden_states[i].float()
            diff = (s_h - t_h) ** 2
            mse = mse + (diff.mean(dim=-1) * mask).sum() / active
            if w_cos > 0:
                s_flat = s_h.view(-1, s_h.size(-1))
                t_flat = t_h.view(-1, t_h.size(-1))
                target = torch.ones(s_flat.size(0), device=s_flat.device)
                cos = cos + F.cosine_embedding_loss(s_flat, t_flat, target)
        mse = mse / max(1, num_layers - 1)
        if w_cos > 0:
            cos = cos / max(1, num_layers - 1)

    total = w_kl * kl + w_ce * ce + w_mse * mse + w_cos * cos
    metrics = {"kl": float(kl.item()), "ce": float(ce.item()), "mse": float(mse.item()), "cosine": float(cos.item())}
    return total, metrics


def _collect_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _sam_apply(params: List[torch.nn.Parameter], rho: float, adaptive: bool) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    grads = []
    shared_device = None
    for p in params:
        if p.grad is None:
            continue
        if shared_device is None:
            shared_device = p.device
        g = p.grad
        if adaptive:
            g = g * p.data.abs()
        grads.append(g)

    if not grads:
        return []

    if shared_device is None:
        shared_device = grads[0].device

    grad_norm = torch.norm(torch.stack([g.norm(p=2).to(shared_device) for g in grads]))
    scale = rho / (grad_norm + 1e-12)

    epsilons: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            current_scale = scale.to(p.device)
            if adaptive:
                eps = (p.grad * p.data.abs()) * current_scale
            else:
                eps = p.grad * current_scale
            p.add_(eps)
            epsilons.append((p, eps))
    return epsilons


def _sam_restore(epsilons: List[Tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for p, eps in epsilons:
            p.sub_(eps)


def _build_text_stream(data_cfg: dict) -> Iterable[str]:
    ds = load_dataset(
        data_cfg["dataset"],
        data_cfg.get("subset"),
        split=data_cfg.get("split", "train"),
        streaming=data_cfg.get("streaming", True),
    )
    min_len = int(data_cfg.get("min_text_length", 50))
    for sample in ds:
        text = sample.get("text", "")
        if isinstance(text, str) and len(text) >= min_len:
            yield text


def _build_batcher(tokenizer: AutoTokenizer, data_cfg: dict, batch_size: int):
    stream = _build_text_stream(data_cfg)
    max_length = int(data_cfg.get("max_length", 4096))

    def get_batch():
        buffer = []
        while len(buffer) < batch_size:
            try:
                buffer.append(next(stream))
            except StopIteration:
                break
        return tokenizer(buffer, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    return get_batch


def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("repro_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _log_disk_snapshot(logger: logging.Logger, path: str) -> None:
    try:
        usage = os.statvfs(path)
        total = usage.f_frsize * usage.f_blocks
        avail = usage.f_frsize * usage.f_bavail
        used = total - avail
        logger.info(
            "disk_snapshot path=%s used_gb=%.2f avail_gb=%.2f total_gb=%.2f",
            path,
            used / (1024**3),
            avail / (1024**3),
            total / (1024**3),
        )
    except Exception:
        logger.warning("failed to collect disk snapshot for %s", path)


def _safe_torch_save(obj, path: str, logger: logging.Logger, retries: int = 2) -> None:
    tmp_path = f"{path}.tmp"
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            elapsed = time.time() - t0
            size_gb = os.path.getsize(path) / (1024**3)
            logger.info(
                "saved_file path=%s size_gb=%.3f elapsed_sec=%.2f attempt=%d",
                path,
                size_gb,
                elapsed,
                attempt,
            )
            return
        except Exception as exc:
            last_exc = exc
            logger.error("save_failed path=%s attempt=%d error=%s", path, attempt, exc)
            logger.error(traceback.format_exc())
            _log_disk_snapshot(logger, os.path.dirname(path) or ".")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    raise RuntimeError(f"failed to save {path} after {retries} attempts: {last_exc}")



def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    if os.path.isfile(checkpoint_path):
        state_dict = load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    for index_name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
        index_path = os.path.join(checkpoint_path, index_name)
        if os.path.exists(index_path):
            import json

            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            shard_files = sorted(set(index_data.get("weight_map", {}).values()))
            for shard_file in shard_files:
                shard_path = os.path.join(checkpoint_path, shard_file)
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Missing shard file: {shard_path}")
                shard_state = load_state_dict(shard_path)
                model.load_state_dict(shard_state, strict=False)
            return

    for weights_name in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
        weights_path = os.path.join(checkpoint_path, weights_name)
        if os.path.exists(weights_path):
            state_dict = load_state_dict(weights_path)
            model.load_state_dict(state_dict, strict=False)
            return

    raise FileNotFoundError(f"No model weights found in checkpoint: {checkpoint_path}")


def _save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    logger: logging.Logger,
) -> None:
    os.makedirs(path, exist_ok=True)
    t0 = time.time()
    logger.info("save_checkpoint_start step=%d path=%s", global_step, path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    _safe_torch_save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        },
        os.path.join(path, "training_state.pt"),
        logger=logger,
        retries=2,
    )
    logger.info("save_checkpoint_done step=%d path=%s elapsed_sec=%.2f", global_step, path, time.time() - t0)


def _load_training_state(
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> int:
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(state_path):
        print("[Resume] training_state.pt not found, resume from model weights only.")
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state.get("global_step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repro Llama3 poly training (aligned with mpcFriendlyLlama2 config).")
    parser.add_argument("--config", default="llama/configs/llama3.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    model_cfg = cfg["model"]
    teacher_cfg = cfg["teacher"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    poly_cfg = cfg["poly"]
    loss_cfg = cfg["loss"]

    tokenizer_id = args.resume_from_checkpoint or model_cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    if poly_cfg.get("enabled", True):
        student = apply_quadratic_poly(student, replace_odd_layers=poly_cfg.get("replace_odd_layers", True))

    if args.resume_from_checkpoint:
        _load_checkpoint_weights(student, args.resume_from_checkpoint)

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_cfg["model_id"],
        device_map=teacher_cfg.get("device_map", "auto"),
        torch_dtype=getattr(torch, teacher_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=teacher_cfg.get("trust_remote_code", True),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if train_cfg.get("gradient_checkpointing", True):
        if hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()
        if hasattr(student, "config") and hasattr(student.config, "use_cache"):
            student.config.use_cache = False

    lr = float(train_cfg.get("lr", 1.5e-5))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    steps = int(train_cfg.get("steps", 5000))
    grad_accum = int(train_cfg.get("grad_accum", 64))
    batch_size = int(train_cfg.get("batch_size", 1))
    max_norm = float(train_cfg.get("max_norm", 0.5))
    log_interval = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 500))
    out_dir = train_cfg.get("output_dir", "outputs/llama3_8b_poly_act")
    os.makedirs(out_dir, exist_ok=True)
    logger = _setup_logger(out_dir)
    logger.info("train_start config=%s", args.config)
    _log_disk_snapshot(logger, out_dir)

    start_step = 0
    if args.resume_from_checkpoint:
        start_step = _load_training_state(args.resume_from_checkpoint, optimizer, scheduler)
        logger.info("resume_loaded start_step=%d checkpoint=%s", start_step, args.resume_from_checkpoint)

    opt_cfg = train_cfg.get("optimizer", {})
    use_sam = bool(opt_cfg.get("use_sam", True))
    sam_rho = float(opt_cfg.get("rho", 0.03))
    sam_adaptive = bool(opt_cfg.get("adaptive", False))
    trainable_params = _collect_trainable_params(student)

    get_batch = _build_batcher(tokenizer, data_cfg, batch_size=batch_size)
    loss_weights = loss_cfg.get("weights", {"kl": 1.0, "ce": 1.0, "mse": 1.0, "cosine": 1.0})

    student.train()
    for step in range(start_step, steps):
        optimizer.zero_grad()
        total_loss = 0.0
        metrics = {}

        for _ in range(grad_accum):
            batch = get_batch()
            if not batch["input_ids"].numel():
                break

            input_ids = batch["input_ids"].to(student.device)
            attention_mask = batch["attention_mask"].to(student.device)

            with torch.no_grad():
                teacher_out = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            if use_sam:
                student_out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                loss, metrics = compute_logic_distill_loss(student_out, teacher_out, attention_mask, input_ids, loss_weights)
                loss = loss / grad_accum
                loss.backward()

                epsilons = _sam_apply(trainable_params, sam_rho, sam_adaptive)
                optimizer.zero_grad()

                student_out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                loss, metrics = compute_logic_distill_loss(student_out, teacher_out, attention_mask, input_ids, loss_weights)
                loss = loss / grad_accum
                loss.backward()
                _sam_restore(epsilons)
                total_loss += loss.item() * grad_accum
            else:
                student_out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                loss, metrics = compute_logic_distill_loss(student_out, teacher_out, attention_mask, input_ids, loss_weights)
                loss = loss / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]) if metrics else ""
            logger.info("step=%d loss=%.4f %s", step, total_loss, metric_str)

        if (step + 1) % save_steps == 0:
            ckpt_dir = os.path.join(out_dir, f"checkpoint-{step + 1}")
            try:
                _save_checkpoint(student, tokenizer, ckpt_dir, optimizer, scheduler, step + 1, logger)
            except Exception as exc:
                logger.error("checkpoint_save_failed step=%d path=%s error=%s", step + 1, ckpt_dir, exc)
                logger.error(traceback.format_exc())
                logger.warning("continue_training_without_state_save step=%d", step + 1)

    try:
        _save_checkpoint(student, tokenizer, out_dir, optimizer, scheduler, steps, logger)
    except Exception as exc:
        logger.error("final_checkpoint_save_failed path=%s error=%s", out_dir, exc)
        logger.error(traceback.format_exc())
    logger.info("train_done output_dir=%s steps=%d", out_dir, steps)
    print({"status": "done", "output_dir": out_dir, "steps": steps})


if __name__ == "__main__":
    main()
