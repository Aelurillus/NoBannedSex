# Local data preparation

There are two ingestion routes. A text/JSONL file passed to `train --data` is a
small in-memory convenience route. A prepared directory is a checked, read-only,
memory-mapped corpus. Neither route downloads datasets or weights.

## Authorized sources

Create a JSON file with `sources`, each containing `path`, `license`, and `source`.
Relative paths resolve against that JSON file. Record actual usage rights; an
unknown/placeholder license is rejected. This check records your assertion; it
does not verify the assertion or grant rights. The sample source manifest points
only to the project-owned synthetic fixture.

Text files use blank-line-separated documents; JSONL requires one `{"text": "..."}`
object per line. UTF-8 decoding is strict. Default maximum document size is one
million characters. Oversized documents fail rather than silently truncate.

```bash
python -m nobannedsex.data --sources /persistent/sources.json \
  --tokenizer /persistent/tokenizer.json --out /persistent/prepared \
  --val-fraction 0.02 --seed 42 --shard-tokens 1000000
```

The tokenizer is optional for byte-based experiments. Train a BPE only on an
approved training corpus, keeping evaluation data separate:

```bash
python -m nobannedsex.tokenizer --files /persistent/train-only.txt \
  --out /persistent/tokenizer.json --vocab-size 32000
```

BPE can produce fewer tokens than requested on small data. Use the actual reported
vocabulary size in the model config. Do not use the sample corpus to construct a
production tokenizer. The preparation tool splits model-training documents; it
does not automatically construct a separate tokenizer-training subset.

## Preparation contract

- Normalize Unicode to NFC, convert newlines, and strip document boundaries.
- Hash normalized UTF-8 documents and globally deduplicate through a disk-backed
  SQLite primary key. No topic filtering or semantic classifier is applied.
- Hash `(seed, document hash)` for stable train/validation membership. Input
  order does not change membership. Very small corpora can yield an empty split;
  preparation fails and asks for more data/a different seed.
- Append EOS between documents and pack into little-endian uint32 token shards.
  A sequence can cross a document/shard boundary. Attention is not reset at EOS;
  this is packed next-token training, not isolated document attention.
- Write the manifest last, including source checksums, source rights, tokenizer
  checksum, normalization version, split recipe, counts and per-shard checksums.
- Retain `documents.sqlite` for hash-based audit/overlap analysis. It is not
  needed by training; source text is not copied into the index.

Memory use is bounded by one parsed document, one shard buffer, SQLite cache and
sampled batches rather than total token count. Mapping metadata/file handles
still grow with shard count; this is a single-host pipeline, not a distributed
object-store loader. Source hashing and full shard integrity checks cost one
sequential read each. The loader checks token bounds and shard sizes/checksums
before use, samples windows across shard boundaries and converts only those
windows to int64 tensors.

`train --data /persistent/prepared` requires the exact tokenizer and matching
split seed/fraction in the training config. Resume checks the manifest fingerprint
and rechecks shard integrity. Local raw-file training deduplicates normalized
documents and uses a seeded shuffled document split; its split algorithm differs
from the stable hash assignment of prepared data. Choose one route per experiment.

## Limits and next steps

Exact deduplication does not detect paraphrases, copied fragments or near
duplicates. The pipeline does not automatically check a separate QA benchmark
for contamination; compare the audit hashes and inspect overlap before reporting
quality scores. Never place held-out tests in tokenizer/pretraining/post-training
inputs. No web collection, third-party dataset download, semantic cleaning or
legal review is implemented. Keep large corpora, `.bin` shards and SQLite indexes
outside Git.
