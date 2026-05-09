"""Bangla tokenizer wrapper for HF Trainer compatibility.

Wraps the raw SentencePiece model so it works with HF Trainer,
DataCollatorForLanguageModeling, and SFTTrainer without broken
LlamaTokenizer conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sentencepiece as spm
from transformers import PreTrainedTokenizer


class BanglaTokenizer(PreTrainedTokenizer):
    """HF-compatible tokenizer wrapping a SentencePiece BPE model."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        sp_model_path: str,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs,
    ):
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(sp_model_path))
        self._sp_model_path = str(sp_model_path)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def get_vocab(self) -> dict[str, int]:
        vocab = {self._sp.IdToPiece(i): i for i in range(self._sp.GetPieceSize())}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return self._sp.EncodeAsPieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._sp.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._sp.IdToPiece(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._sp.DecodePieces(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        out_path = save_dir / f"{prefix}bangla_bpe_32k.model"

        # Copy the sp model file
        import shutil
        if str(out_path) != self._sp_model_path:
            shutil.copy2(self._sp_model_path, out_path)

        return (str(out_path),)


def load_bangla_tokenizer(model_dir: str) -> BanglaTokenizer:
    """Load BanglaTokenizer from a model directory or SP model path.

    Looks for bangla_bpe_32k.model in the directory, or uses the path directly.
    """
    model_dir = Path(model_dir)

    # Check common locations
    candidates = [
        model_dir / "bangla_bpe_32k.model",
        model_dir / "tokenizer" / "output" / "bangla_bpe_32k.model",
        model_dir,  # direct path to .model file
    ]

    for candidate in candidates:
        if candidate.is_file() and str(candidate).endswith(".model"):
            tokenizer = BanglaTokenizer(sp_model_path=str(candidate))

            # Restore any added special tokens saved alongside the checkpoint.
            checkpoint_dir = candidate.parent if candidate.name.endswith(".model") else model_dir
            added_tokens_path = checkpoint_dir / "added_tokens.json"
            tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"

            extra_tokens: list[str] = []

            if added_tokens_path.exists():
                with open(added_tokens_path, encoding="utf-8") as fh:
                    added_tokens = json.load(fh)
                extra_tokens.extend(sorted(added_tokens, key=added_tokens.get))

            if tokenizer_config_path.exists():
                with open(tokenizer_config_path, encoding="utf-8") as fh:
                    tokenizer_config = json.load(fh)
                config_extra = tokenizer_config.get("extra_special_tokens", [])
                if isinstance(config_extra, list):
                    extra_tokens.extend(config_extra)

            if extra_tokens:
                deduped_tokens = list(dict.fromkeys(extra_tokens))
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": deduped_tokens}
                )

            return tokenizer

    raise FileNotFoundError(
        f"Could not find bangla_bpe_32k.model in {model_dir}. "
        f"Searched: {[str(c) for c in candidates]}"
    )
