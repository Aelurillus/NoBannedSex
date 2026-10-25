import hashlib
import json
import sqlite3

import numpy as np
import pytest
import torch

from nobannedsex.checkpoint import load_checkpoint
from nobannedsex.data import document_split, iter_documents, open_dataset, prepare_dataset
from nobannedsex.tokenizer import ByteTokenizer
from nobannedsex.train import sample_batch, train


@pytest.fixture
def source_manifest(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    docs = [f"Unique document {i}: " + "engineering token shard test. " * 5 for i in range(80)]
    docs += [docs[0], "café", "cafe\u0301"]
    corpus.write_text("\n".join(json.dumps({"text": d}) for d in docs), encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{"path": corpus.name, "license": "Apache-2.0",
                                              "source": "project-owned synthetic test fixture"}]}))
    return sources


def test_shards_dedup_mmap_and_cross_boundary(source_manifest, tmp_path):
    root = tmp_path / "prepared"
    prepare_dataset(source_manifest, root, shard_tokens=23, val_fraction=.15)
    tk = ByteTokenizer()
    training, validation, manifest, _ = open_dataset(root, tk, 16)
    assert manifest["duplicates_removed"] == 2
    assert sum(manifest["documents"].values()) == 81
    for split in (training, validation):
        assert all(isinstance(m, np.memmap) for m in split.maps)
        reference = np.concatenate(split.maps)
        for start in (0, 21, 22, 23, len(split) - 17):
            np.testing.assert_array_equal(split.window(start, 17), reference[start:start + 17])
        x, y = sample_batch(split, 3, 16, torch.device("cpu"), torch.Generator().manual_seed(9))
        assert x.shape == y.shape == (3, 16)
        assert torch.equal(x[:, 1:], y[:, :-1])
    with sqlite3.connect(root / "documents.sqlite") as db:
        rows = db.execute("SELECT sha256, split FROM documents").fetchall()
        assert len(rows) == len({r[0] for r in rows})
        for digest, split in rows:
            assert document_split(digest, 42, .15) == split


def test_split_membership_independent_of_source_order(source_manifest, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_dataset(source_manifest, first)
    corpus = source_manifest.parent / "corpus.jsonl"
    corpus.write_text("\n".join(reversed(corpus.read_text().splitlines())))
    prepare_dataset(source_manifest, second)
    def index(root):
        with sqlite3.connect(root / "documents.sqlite") as db:
            return dict(db.execute("SELECT sha256, split FROM documents"))
    assert index(first) == index(second)


def test_corrupted_shard_rejected(source_manifest, tmp_path):
    root = tmp_path / "prepared"
    prepare_dataset(source_manifest, root)
    manifest = json.loads((root / "manifest.json").read_text())
    shard = root / manifest["splits"]["train"][0]["path"]
    with shard.open("r+b") as f:
        f.write(b"\xff\xff\xff\xff")
    with pytest.raises(ValueError, match="checksum"):
        open_dataset(root, ByteTokenizer(), 16)


def test_tokenizer_fingerprint_recorded(source_manifest, tmp_path):
    root = tmp_path / "prepared"
    prepare_dataset(source_manifest, root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["tokenizer_sha256"] is None
    corpus = source_manifest.parent / "corpus.jsonl"
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()


def test_source_rights_required(source_manifest, tmp_path):
    sources = json.loads(source_manifest.read_text())
    sources["sources"][0]["license"] = "unknown"
    source_manifest.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="usage rights"):
        prepare_dataset(source_manifest, tmp_path / "prepared")


def test_text_and_jsonl_readers_and_document_limit(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text("first\nline\n\nsecond\n")
    assert list(iter_documents(text)) == ["first\nline", "second"]
    with pytest.raises(ValueError, match="exceeds"):
        list(iter_documents(text, max_document_chars=4))
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text('{"text": 123}\n')
    with pytest.raises(ValueError, match="string text"):
        list(iter_documents(jsonl))


def test_mmap_training_resume(source_manifest, training_case, tmp_path):
    args, _ = training_case
    root = tmp_path / "prepared"
    prepare_dataset(source_manifest, root, val_fraction=.15, shard_tokens=31)
    args.data = str(root)
    full = load_checkpoint(train(args))
    args.out = str(tmp_path / "interrupted-mmap")
    args.stop_after = 2
    cp = train(args)
    args.resume, args.stop_after = str(cp), None
    recovered = load_checkpoint(train(args))
    for name in full["model"]:
        torch.testing.assert_close(full["model"][name], recovered["model"][name], atol=0, rtol=0)


def test_shard_path_cannot_escape_root(source_manifest, tmp_path):
    root = tmp_path / "prepared"
    prepare_dataset(source_manifest, root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"][0]["path"] = "../elsewhere.bin"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        open_dataset(root, ByteTokenizer(), 16)
