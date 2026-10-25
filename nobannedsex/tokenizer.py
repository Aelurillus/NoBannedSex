"""Deterministic byte tokenizer for smoke tests; trainable byte-level BPE for serious experiments."""
from __future__ import annotations

import argparse
from pathlib import Path


class ByteTokenizer:
    # 0..255 are UTF-8 bytes, 256..259 are reserved special tokens.
    pad_id = 256
    bos_id = 257
    eos_id = 258
    unk_id = 259
    vocab_size = 260

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        return ([self.bos_id] if bos else []) + list(text.encode("utf-8")) + ([self.eos_id] if eos else [])

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i <= 255).decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError('Install the optional dependency: pip install -e ".[tokenizer]"') from exc
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()
        self.pad_id = self.tk.token_to_id("<pad>")
        self.bos_id = self.tk.token_to_id("<bos>")
        self.eos_id = self.tk.token_to_id("<eos>")
        self.unk_id = self.tk.token_to_id("<unk>")
        if any(x is None for x in (self.pad_id, self.bos_id, self.eos_id, self.unk_id)):
            raise ValueError("tokenizer missing required special tokens")

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        return ([self.bos_id] if bos else []) + self.tk.encode(text).ids + ([self.eos_id] if eos else [])

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)


def load_tokenizer(path: str | None):
    return BPETokenizer(path) if path else ByteTokenizer()


def train_bpe(files: list[str], output: str, vocab_size: int = 32000) -> None:
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError as exc:
        raise RuntimeError('Install with: pip install -e ".[tokenizer]"') from exc
    if vocab_size < 300:
        raise ValueError("BPE vocab_size must be >= 300")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train(files, trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tok.save(output)
    print(f"Saved BPE tokenizer ({tok.get_vocab_size()} tokens): {output}")


def main():
    p = argparse.ArgumentParser(description="Train a new byte-level BPE tokenizer from your authorized corpus")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out", default="artifacts/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()
    train_bpe(args.files, args.out, args.vocab_size)


if __name__ == "__main__":
    main()
