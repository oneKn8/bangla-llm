# Bangla-LLM

A 306M-parameter Bengali language model trained from scratch on a curated Bengali text corpus. LLaMA-style decoder-only transformer with GQA, SwiGLU, RMSNorm, and RoPE.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 306M |
| Layers | 18 |
| Hidden dim | 1024 |
| Attention heads | 16 (4 KV heads, GQA) |
| FFN dim | 4096 (SwiGLU) |
| Vocab size | 32,000 (BPE) |
| Context length | 2,048 |
| Precision | BF16 |
| Weight tying | Yes |

## Training Data

~676M tokens (2.03B effective over 3 epochs) from curated Bengali sources:

| Source | Size | Docs |
|--------|------|------|
| Culturax Bengali | 3.7 GB | ~545K |
| Bengali Wikipedia | 1.2 GB | ~130K |
| Sangraha (AI4Bharat) | 3.3 GB | ~500K |
| **Total (merged, deduped)** | **8.0 GB** | **~1.2M** |

All data passes through a multi-stage cleaning pipeline: HTML normalization, NFKC + Bengali-specific unicode normalization (bnorm), fasttext language detection, quality filtering, and two-pass deduplication (SHA-256 exact + MinHash LSH near-dedup).

Zero tolerance for Hindi/Devanagari contamination -- enforced at both the data pipeline and tokenizer audit stages.

## Tokenizer

32K BPE tokenizer trained with SentencePiece on the cleaned corpus. Bengali-optimized with byte fallback for mixed-script content (Banglish, URLs). Identity normalization (corpus is pre-normalized). Vocabulary audited for Indic script contamination.

## Project Structure

```
bangla-llm/
├── data-pipeline/          # Data collection and processing
│   ├── collectors/         # Source-specific collectors
│   │   ├── wikipedia.py    #   Bengali Wikipedia + Wikisource
│   │   ├── newspaper.py    #   Prothom Alo, Kaler Kantho, Ittefaq
│   │   ├── hf_corpus.py    #   OSCAR, Sangraha, CC-100
│   │   ├── literature.py   #   Internet Archive
│   │   ├── web_crawl.py    #   BFS web crawler
│   │   └── banglish.py     #   Reddit r/bangladesh
│   ├── processing/         # Normalization, lang detect, quality, dedup
│   ├── collect.py          # CLI orchestrator
│   ├── merge_all.py        # Cross-source dedup and merge
│   └── export.py           # Export to training format
├── tokenizer/
│   ├── train_tokenizer.py  # Train 32K BPE tokenizer
│   └── audit_vocab.py      # Script contamination scanner
├── training/
│   ├── config.py           # Model hyperparameters (BanglaLLMConfig)
│   ├── dataset.py          # Binary token dataset + tokenization
│   └── train.py            # Pre-training with HF Accelerate
├── finetune/
│   ├── sft.py              # Supervised fine-tuning (full + LoRA)
│   ├── create_sft_data.py  # Prepare instruction data
│   └── generate.py         # Inference / text generation
├── scripts/
│   ├── param_count.py      # Architecture parameter calculator
│   └── estimate_tokens.py  # Corpus token estimator
└── docs/
    └── plans/              # Build plans and design docs
```

## Training

### Pre-training

Trained on a single A100 80GB using HuggingFace Accelerate.

```
Optimizer:      AdamW (lr=3e-4, cosine decay to 3e-5)
Warmup:         2,000 steps
Batch size:     8 micro x 32 grad accum = 256 sequences (524K tokens)
Total steps:    3,870
Checkpoints:    Every 1,000 steps
Hardware:       1x NVIDIA A100 80GB
Time:           ~10 hours
```

### Fine-tuning

Supervised fine-tuning with TRL's SFTTrainer. Supports both full fine-tuning and LoRA adapters. ChatML format with `<|im_start|>` / `<|im_end|>` tokens.

A news-domain LoRA variant is trained separately on Bengali newspaper data.

## Evaluation

Benchmarks (planned):
- Bengali perplexity on held-out Wikipedia / newspaper test sets
- BanglaNLG generation quality
- Comparison against multilingual baselines (BLOOM-560M, mGPT)
- Tokenizer efficiency vs LLaMA / GPT tokenizers on Bengali text

## Usage

### Generate text

```bash
python finetune/generate.py \
    --model training/checkpoints/final \
    --prompt "বাংলাদেশের রাজধানী" \
    --max-tokens 200
```

### Fine-tune with LoRA

```bash
python finetune/sft.py \
    --base-model training/checkpoints/final \
    --data finetune/data/sft_train.jsonl \
    --lora \
    --epochs 3 \
    --lr 2e-5
```

## Data Pipeline

The data pipeline handles collection, cleaning, and deduplication:

```bash
# Collect from a source
python data-pipeline/collect.py wikipedia

# Process raw data (normalize, filter, dedup)
python data-pipeline/pipeline.py

# Merge all sources with cross-source dedup
python data-pipeline/merge_all.py

# Export to training-ready binary tokens
python training/dataset.py --tokenize
```

Quality gates at each stage:
- **Language detection:** fasttext lid.176.bin, Bengali score >= 0.65, Hindi/Assamese < 0.2
- **Script filtering:** Devanagari char ratio < 1%, zero Assamese-only chars
- **Quality:** Min 100 chars, >= 50% Bengali script ratio, <= 30% repeat lines
- **Dedup:** SHA-256 exact match + MinHash LSH (Jaccard >= 0.8)

## Why

This is the first from-scratch Bengali language model built with a fully documented, reproducible pipeline. Most existing Bengali NLP relies on multilingual models (mBERT, XLM-R, BLOOM) that allocate a fraction of their capacity to Bengali. A dedicated model with a Bengali-optimized tokenizer and clean monolingual data achieves better token efficiency and downstream performance at a fraction of the size.

## License

TBD

## Citation

```bibtex
@misc{bangla-llm-2026,
    title={Bangla-LLM: A 306M Parameter Bengali Language Model},
    author={Shifat Islam Santo},
    year={2026}
}
```
