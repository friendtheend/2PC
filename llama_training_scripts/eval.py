from __future__ import annotations

import re

import torch
from datasets import load_dataset


def format_arc_prompt(sample: dict) -> str:
    question = sample["question"]
    choices = sample["choices"]
    lines = [f"Question: {question}", "Choices:"]
    for label, text in zip(choices["label"], choices["text"]):
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_answer(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    if match:
        return match.group(1)
    for char in text.strip().upper():
        if char in {"A", "B", "C", "D"}:
            return char
    return None


def _choice_loglikelihood(model, tokenizer, prompt: str, choice_label: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    choice_ids = tokenizer(choice_label, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    input_ids = torch.cat([prompt_ids, choice_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    start = prompt_ids.shape[1] - 1
    end = input_ids.shape[1] - 1
    token_ll = 0.0
    for t in range(start, end):
        token_id = target_ids[0, t].item()
        token_ll += log_probs[0, t, token_id].item()
    return token_ll


def _evaluate_generate(model, tokenizer, dataset, cfg: dict) -> tuple[int, int]:
    max_samples = int(cfg.get("max_samples", 1000))
    max_new_tokens = int(cfg.get("max_new_tokens", 4))
    do_sample = bool(cfg.get("do_sample", False))
    temperature = float(cfg.get("temperature", 0.1))

    correct = 0
    total = 0

    for sample in dataset:
        prompt = format_arc_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        answer_text = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_answer(answer_text) or ""
        if pred == sample["answerKey"]:
            correct += 1
        total += 1
        if total >= max_samples:
            break

    return correct, total


def _evaluate_loglikelihood(model, tokenizer, dataset, cfg: dict) -> tuple[int, int]:
    max_samples = int(cfg.get("max_samples", 1000))
    correct = 0
    total = 0

    for sample in dataset:
        prompt = format_arc_prompt(sample)
        labels = sample["choices"]["label"]

        scores = {}
        for label in labels:
            scores[label] = _choice_loglikelihood(model, tokenizer, prompt, label)

        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        if pred == sample["answerKey"]:
            correct += 1
        total += 1
        if total >= max_samples:
            break

    return correct, total


def evaluate_arc_easy(model, tokenizer, cfg: dict) -> dict[str, float | str]:
    dataset = load_dataset(
        cfg.get("dataset_name", "allenai/ai2_arc"),
        cfg.get("subset", "ARC-Easy"),
        split=cfg.get("split", "test"),
        cache_dir=cfg.get("cache_dir"),
    )

    method = str(cfg.get("method", "loglikelihood")).lower()
    model.eval()

    if method == "generate":
        correct, total = _evaluate_generate(model, tokenizer, dataset, cfg)
    elif method == "loglikelihood":
        correct, total = _evaluate_loglikelihood(model, tokenizer, dataset, cfg)
    else:
        raise ValueError("Unsupported evaluation.method. Use 'loglikelihood' or 'generate'.")

    acc = correct / max(1, total)
    return {
        "method": method,
        "accuracy": acc,
        "correct": float(correct),
        "total": float(total),
    }
