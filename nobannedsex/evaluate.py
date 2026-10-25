"""Deterministic LM and short-answer evaluation. Fixtures test tooling, not quality."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import file_sha256
from .data import TokenSplit, iter_documents
from .generate import load_bundle


def normalized_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold()).rstrip(".。!?！？")


ABSTENTIONS = {"unknown", "i don't know", "insufficient information", "不知道", "信息不足"}


def read_suite(path):
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("suite requires unique string IDs")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("suite prompt must be nonempty")
        if not isinstance(row.get("domain"), str) or type(row.get("answerable")) is not bool:
            raise ValueError("suite requires domain and boolean answerable")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("suite requires nonempty reference answers")
        if not row["answerable"] and not all(normalized_answer(a) in ABSTENTIONS for a in answers):
            raise ValueError("unanswerable references must be controlled abstention strings")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("suite must not be empty")
    return rows


def score_predictions(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("prediction count mismatch")
    answerable = sum(r["answerable"] for r in rows)
    unknown = len(rows) - answerable
    correct = answered = wrong = abstained_known = abstained_unknown = empty_known = empty_unknown = 0
    details = []
    for row, prediction in zip(rows, predictions):
        normalized = normalized_answer(prediction)
        abstained = normalized in ABSTENTIONS
        empty = not normalized
        match = normalized in {normalized_answer(a) for a in row["answers"]}
        if row["answerable"]:
            correct += match
            answered += not abstained and not empty
            wrong += not abstained and not empty and not match
            empty_known += empty
            abstained_known += abstained
        else:
            abstained_unknown += abstained
            empty_unknown += empty
        details.append({**row, "prediction": prediction, "exact_match": match, "abstained": abstained, "empty": empty})
    def ratio(a, b):
        return a / b if b else None
    return {
        "count": len(rows), "answerable_count": answerable, "unanswerable_count": unknown,
        "answerable_exact_match": ratio(correct, answerable),
        "answerable_answer_coverage": ratio(answered, answerable),
        "answerable_wrong_answer_rate": ratio(wrong, answerable),
        "answerable_excessive_abstention_rate": ratio(abstained_known, answerable),
        "unanswerable_abstention_rate": ratio(abstained_unknown, unknown),
        "answerable_empty_response_rate": ratio(empty_known, answerable),
        "unanswerable_empty_response_rate": ratio(empty_unknown, unknown),
    }, details


@torch.inference_mode()
def language_model_metrics(model, tokens):
    """Count each next-token target once, including a short final block."""
    model.eval()
    if len(tokens) < 2:
        raise ValueError("LM evaluation needs at least two tokens")
    nll, correct, total = 0.0, 0, 0
    device = next(model.parameters()).device
    for start in range(0, len(tokens) - 1, model.cfg.max_seq_len):
        count = min(model.cfg.max_seq_len, len(tokens) - start - 1)
        if isinstance(tokens, TokenSplit):
            window = torch.from_numpy(tokens.window(start, count + 1))
        elif isinstance(tokens, np.ndarray):
            window = torch.as_tensor(tokens[start:start + count + 1], dtype=torch.long)
        else:
            window = torch.tensor(tokens[start:start + count + 1], dtype=torch.long)
        window = window.to(device).unsqueeze(0)
        logits, _ = model(window[:, :-1])
        target = window[:, 1:]
        nll += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == target).sum().item()
        total += count
    return {"tokens": total, "loss": nll / total, "perplexity": math.exp(min(nll / total, 80)),
            "next_token_accuracy": correct / total}


def run_evaluation(checkpoint, suite, out, device="cpu", max_new_tokens=16, lm_data=None):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch.set_num_threads(4)
    model, tk = load_bundle(str(checkpoint), device)
    rows = read_suite(suite)
    predictions = []
    for row in rows:
        prompt = tk.encode(row["prompt"], bos=True)
        if len(prompt) > model.cfg.max_seq_len:
            raise ValueError(f"suite prompt {row['id']} exceeds model context; no silent truncation")
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        output = model.generate(ids, max_new_tokens, temperature=0, eos_id=tk.eos_id)
        predictions.append(tk.decode(output[0, len(prompt):].tolist()))
    metrics, details = score_predictions(rows, predictions)
    report = {
        "format_version": 1, "checkpoint_sha256": file_sha256(checkpoint),
        "suite_sha256": file_sha256(suite), "device": device, "torch_version": str(torch.__version__),
        "decoding": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "limitations": "Controlled short-answer exact match and fixed abstention strings only; no semantic judge, confidence estimate, or validated capability benchmark. Synthetic smoke fixtures are tooling checks.",
        "metrics": metrics, "examples": details,
        "by_domain": {domain: score_predictions([r for r in rows if r["domain"] == domain],
                                                [p for r, p in zip(rows, predictions) if r["domain"] == domain])[0]
                      for domain in sorted({r["domain"] for r in rows})},
    }
    if lm_data:
        tokens = [t for doc in iter_documents(lm_data) for t in tk.encode(doc, eos=True)]
        report["language_model"] = {**language_model_metrics(model, tokens), "data_sha256": file_sha256(lm_data)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--lm-data", help="small held-out text/JSONL only (loaded into RAM)")
    args = p.parse_args()
    report = run_evaluation(args.checkpoint, args.suite, args.out, args.device, args.max_new_tokens, args.lm_data)
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
