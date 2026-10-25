import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nobannedsex.config import ModelConfig
from nobannedsex.evaluate import language_model_metrics, read_suite, run_evaluation, score_predictions
from nobannedsex.model import NoBannedSexLM
from nobannedsex.train import train


ROOT = Path(__file__).resolve().parents[1]


def test_accuracy_and_answer_coverage_separate():
    rows = [
        {"id": "a", "answerable": True, "answers": ["Paris"]},
        {"id": "b", "answerable": True, "answers": ["5"]},
        {"id": "c", "answerable": False, "answers": ["unknown"]},
    ]
    metrics, _ = score_predictions(rows, ["Paris.", "unknown", "unknown"])
    assert metrics["answerable_exact_match"] == .5
    assert metrics["answerable_answer_coverage"] == .5
    assert metrics["answerable_excessive_abstention_rate"] == .5
    assert metrics["unanswerable_abstention_rate"] == 1
    zero, _ = score_predictions(rows, ["unknown"] * 3)
    assert zero["answerable_exact_match"] == 0
    assert zero["answerable_answer_coverage"] == 0
    assert zero["unanswerable_abstention_rate"] == 1


def test_lm_loss_counts_partial_last_block():
    model = NoBannedSexLM(ModelConfig(dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                      hidden_dim=32, max_seq_len=4)).eval()
    tokens = list(range(11))
    metrics = language_model_metrics(model, tokens)
    assert metrics["tokens"] == 10
    total = 0
    for start in (0, 4, 8):
        block = torch.tensor([tokens[start:min(start + 5, len(tokens))]])
        logits, _ = model(block[:, :-1])
        total += F.cross_entropy(logits.reshape(-1, 260), block[:, 1:].reshape(-1), reduction="sum").item()
    assert metrics["loss"] == pytest.approx(total / 10)


def test_evaluation_writes_reproducible_report(training_case, tmp_path):
    args, config = training_case
    config["model"]["max_seq_len"] = 64
    args.stop_after = 1
    Path(args.config).write_text(json.dumps(config))
    cp = train(args)
    report = run_evaluation(cp, ROOT / "data/eval-smoke.jsonl", tmp_path / "report.json", max_new_tokens=3,
                            lm_data=ROOT / "data/sample.txt")
    assert report["metrics"]["count"] == 4
    assert report["language_model"]["tokens"] > 0
    assert "Synthetic smoke" in report["limitations"]
    second = run_evaluation(cp, ROOT / "data/eval-smoke.jsonl", tmp_path / "report-2.json", max_new_tokens=3,
                            lm_data=ROOT / "data/sample.txt")
    assert report == second


def test_eval_rejects_truncated_prompt(training_case, tmp_path):
    args, _ = training_case
    args.stop_after = 1
    cp = train(args)
    with pytest.raises(ValueError, match="exceeds model context"):
        run_evaluation(cp, ROOT / "data/eval-smoke.jsonl", tmp_path / "report.json")


def test_suite_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "suite.jsonl"
    row = {"id": "duplicate", "domain": "math", "prompt": "1+1", "answerable": True, "answers": ["2"]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="unique"):
        read_suite(path)


def test_empty_response_is_not_answer_or_abstention():
    rows = [{"id": "a", "answerable": True, "answers": ["Paris"]},
            {"id": "b", "answerable": False, "answers": ["unknown"]}]
    metrics, _ = score_predictions(rows, ["", "  "])
    assert metrics["answerable_answer_coverage"] == 0
    assert metrics["answerable_empty_response_rate"] == 1
    assert metrics["unanswerable_empty_response_rate"] == 1
    assert metrics["unanswerable_abstention_rate"] == 0
