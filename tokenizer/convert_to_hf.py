"""Convert SentencePiece tokenizer to HuggingFace format and save into model checkpoint dirs."""

from pathlib import Path
from transformers import LlamaTokenizerFast
import sentencepiece as spm

SP_MODEL = "tokenizer/output/bangla_bpe_32k.model"
CHECKPOINTS = [
    "training/checkpoints/step-4000",
    "training/checkpoints/step-3000",
    "training/checkpoints/epoch-1",
]

print(f"Loading SentencePiece model: {SP_MODEL}")
sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)
print(f"Vocab size: {sp.GetPieceSize()}")

tokenizer = LlamaTokenizerFast(
    vocab_file=SP_MODEL,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<unk>",
    clean_up_tokenization_spaces=False,
)

for ckpt in CHECKPOINTS:
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        print(f"SKIP (not found): {ckpt}")
        continue
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"Saved HF tokenizer to: {ckpt}")

print("Done.")
