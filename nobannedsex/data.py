"""Local corpus preparation: normalized exact dedup, stable splits, mmap token shards.

No corpus is downloaded. Only one bounded document and token shard are buffered;
SQLite keeps deduplication hashes on disk. Documents are EOS-separated and packed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

import numpy as np
import torch

from .checkpoint import file_sha256
from .tokenizer import load_tokenizer


MANIFEST_VERSION = 1
NORMALIZATION = "unicode-nfc-newlines-strip-v1"


def normalize_document(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def iter_documents(path: str | Path, max_document_chars: int = 1_000_000):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        paragraph = []
        size = 0
        for line_number, line in enumerate(f, 1):
            if path.suffix.lower() == ".jsonl":
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    raise ValueError(f"{path}:{line_number}: JSONL requires string text")
                text = obj["text"]
                if len(text) > max_document_chars:
                    raise ValueError("document exceeds max_document_chars")
                text = normalize_document(text)
                if text:
                    yield text
            elif line.strip():
                size += len(line)
                if size > max_document_chars:
                    raise ValueError("document exceeds max_document_chars")
                paragraph.append(line)
            elif paragraph:
                yield normalize_document("".join(paragraph))
                paragraph, size = [], 0
        if paragraph:
            yield normalize_document("".join(paragraph))


def document_split(digest: str, seed: int, val_fraction: float) -> str:
    value = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
    return "validation" if int.from_bytes(value[:8], "big") / 2**64 < val_fraction else "train"


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_tokens: int):
        self.root, self.split, self.limit = root, split, shard_tokens
        self.buffer, self.entries = [], []

    def add(self, ids):
        offset = 0
        while offset < len(ids):
            count = min(self.limit - len(self.buffer), len(ids) - offset)
            self.buffer.extend(ids[offset:offset + count])
            offset += count
            if len(self.buffer) == self.limit:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        name = f"{self.split}-{len(self.entries):05d}.bin"
        path = self.root / name
        np.asarray(self.buffer, dtype="<u4").tofile(path)
        self.entries.append({"path": name, "tokens": len(self.buffer), "sha256": file_sha256(path)})
        self.buffer.clear()


def prepare_dataset(sources: str | Path, out: str | Path, tokenizer_path: str | None = None,
                    val_fraction: float = 0.1, seed: int = 42, shard_tokens: int = 1_000_000,
                    max_document_chars: int = 1_000_000) -> Path:
    if not 0 < val_fraction < 1 or shard_tokens < 1 or max_document_chars < 1:
        raise ValueError("invalid split fraction or shard/document size")
    sources = Path(sources).resolve()
    specification = json.loads(sources.read_text(encoding="utf-8"))
    rows = specification.get("sources", [])
    if not rows:
        raise ValueError("source manifest requires a nonempty sources list")
    resolved = []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("path", "license", "source")):
            raise ValueError("each source requires nonempty path, license and source provenance")
        if row["license"].strip().lower() in ("unknown", "todo", "unspecified"):
            raise ValueError("record actual source usage rights before preparing data")
        path = (sources.parent / row["path"]).resolve()
        resolved.append((path, {**row, "sha256": file_sha256(path)}))
    tk = load_tokenizer(tokenizer_path)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: ShardWriter(out, s, shard_tokens) for s in ("train", "validation")}
    counts = {s: 0 for s in writers}
    duplicates = 0
    # Leave the on-disk hash index as an audit/contamination-check artifact.
    with sqlite3.connect(out / "documents.sqlite") as db:
        db.execute("CREATE TABLE documents (sha256 TEXT PRIMARY KEY, split TEXT, source_index INTEGER)")
        for source_index, (path, _) in enumerate(resolved):
            for text in iter_documents(path, max_document_chars):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                split = document_split(digest, seed, val_fraction)
                cursor = db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (digest, split, source_index))
                if cursor.rowcount == 0:
                    duplicates += 1
                    continue
                ids = tk.encode(text, eos=True)
                if not ids or min(ids) < 0 or max(ids) >= tk.vocab_size:
                    raise ValueError("tokenizer produced invalid token IDs")
                writers[split].add(ids)
                counts[split] += 1
        for writer in writers.values():
            writer.flush()
    if not all(counts.values()):
        raise ValueError("hash split produced an empty partition; provide more documents or another seed")
    manifest = {
        "format_version": MANIFEST_VERSION, "dtype": "<u4", "normalization": NORMALIZATION,
        "seed": seed, "val_fraction": val_fraction, "vocab_size": tk.vocab_size,
        "tokenizer_sha256": file_sha256(tokenizer_path) if tokenizer_path else None,
        "sources": [meta for _, meta in resolved], "documents": counts,
        "duplicates_removed": duplicates, "document_index_sha256": file_sha256(out / "documents.sqlite"),
        "splits": {s: w.entries for s, w in writers.items()},
    }
    target = out / "manifest.json"
    # Publishing manifest last makes incomplete preparations unusable by trainer.
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


class TokenSplit:
    def __init__(self, root: Path, entries: list[dict], vocab_size: int):
        self.maps, self.ends = [], []
        total = 0
        for entry in entries:
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("shard path escapes dataset directory")
            count = entry["tokens"]
            if type(count) is not int or count <= 0 or path.stat().st_size != 4 * count:
                raise ValueError("shard size mismatch")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("shard checksum mismatch")
            data = np.memmap(path, dtype="<u4", mode="r", shape=(count,))
            # A bounded chunk check avoids loading large shards into RAM.
            for i in range(0, count, 1_000_000):
                if int(data[i:i + 1_000_000].max()) >= vocab_size:
                    raise ValueError("shard contains out-of-vocabulary token")
            self.maps.append(data)
            total += count
            self.ends.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def window(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 1 or start + count > self.total:
            raise ValueError("token window outside split")
        result = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            index = bisect_right(self.ends, start)
            base = self.ends[index - 1] if index else 0
            take = min(count - written, self.ends[index] - start)
            result[written:written + take] = self.maps[index][start - base:start - base + take]
            start += take
            written += take
        return result

    def batch(self, starts: list[int], count: int) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.window(start, count) for start in starts]))


def open_dataset(root: str | Path, tokenizer, seq_len: int):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != MANIFEST_VERSION or manifest.get("dtype") != "<u4":
        raise ValueError("unsupported dataset format")
    if manifest["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("dataset/tokenizer vocab mismatch")
    splits = [TokenSplit(root, manifest["splits"][s], tokenizer.vocab_size) for s in ("train", "validation")]
    if any(len(s) < seq_len + 1 for s in splits):
        raise ValueError("dataset split too short for configured sequence length")
    return *splits, manifest, file_sha256(manifest_path)


def main():
    p = argparse.ArgumentParser(description="Prepare local authorized documents into mmap token shards")
    p.add_argument("--sources", required=True, help="JSON source/license manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-tokens", type=int, default=1_000_000)
    p.add_argument("--max-document-chars", type=int, default=1_000_000)
    args = p.parse_args()
    print(prepare_dataset(args.sources, args.out, args.tokenizer, args.val_fraction, args.seed,
                          args.shard_tokens, args.max_document_chars))


if __name__ == "__main__":
    main()
