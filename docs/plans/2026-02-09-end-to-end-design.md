# Bangla LLM - End-to-End Design

## Overview

Build a pure Bangla language model from scratch. Decoder-only, LLaMA-style architecture. Start at ~300M parameters, validate end-to-end, then scale to 1B+. Note: with ~5-6GB clean text (~600M-1B tokens), this is over-parameterized by Chinchilla standards, which is normal for low-resource languages. Multi-epoch training (3-4 epochs) and adding Sangraha data will close the gap.

Goals: general-purpose chatbot, NLP foundation model, task-specific fine-tuning, and learning the full training pipeline.

## Six Phases

| Phase | What | Depends On | Estimated Cost |
|-------|------|------------|----------------|
| 1. Data Pipeline | Collect, clean, dedup ~5-6GB Bangla text | Nothing (already planned) | ~$0 (CPU work) |
| 2. Tokenizer | Train 32K BPE on cleaned corpus | Phase 1 complete | ~$0 (CPU, minutes) |
| 3. Pre-training (300M) | Train causal LM from random init | Phases 1+2 | ~$50-150 (A100s) |
| 4. Fine-tuning | SFT + DPO for instruction-following and alignment | Phase 3 | ~$20-50 (A100s) |
| 5. Evaluation | Benchmarks + human eval for Bangla quality | Phase 4 | ~$0-20 |
| 6. Serving/Deployment | Quantize + serve via vLLM or llama.cpp | Phase 5 | Varies |

Total estimated cost for 300M: **$70-220**

---

## Phase 1: Data Pipeline

Already designed in `data-pipeline/PLAN.md`. Summary:

- 11 sources: Wikipedia, Wikisource, Prothom Alo, Kaler Kantho, Ittefaq, Literature, OSCAR, Sangraha, CC-100, Web Crawl, Banglish (Reddit)
- Pipeline: collect -> normalize (NFKC + bnUnicodeNormalizer + csebuetnlp) -> lang_detect (fasttext) -> quality filter -> dedup (SHA-256 + MinHash LSH)
- Output: ~5-6GB clean text, ~3-3.5M docs after cross-source dedup
- Zero tolerance for Hindi/Assamese contamination
- Storage: Google Drive (2 accounts, 4TB total)

---

## Phase 2: Tokenizer Training

**Tool:** SentencePiece (BPE mode)

**Config:**
- Vocab size: 32,000
- Training input: random 1-2GB sample from cleaned corpus (post lang_detect, post quality filter)
- Character coverage: 0.9999
- Byte fallback: enabled
- Normalization: `normalization_rule_name="identity"` (NO normalization by SentencePiece itself). Our pipeline already applies NFKC + bnorm + cleanup, so tokenizer must see text exactly as-is. Using NFKC in SentencePiece would double-normalize and could undo bnorm's nukta recomposition.
- Split on whitespace + digits (individual digit tokens)
- Special tokens: `<s>`, `</s>`, `<unk>`, `<pad>` + 32 reserved slots for fine-tuning tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, etc.)

**Contamination prevention:**
- Training sample comes from already-cleaned corpus only
- Post-training vocab audit script scans all 32K tokens:
  - Flag/remove tokens containing Devanagari (U+0900-U+097F)
  - Flag/remove tokens containing Assamese-only chars (U+09F0, U+09F1)
  - Flag/remove tokens containing Odia, Tamil, Telugu, Kannada, Malayalam, Gurmukhi, Gujarati ranges
  - Latin characters (a-z, A-Z) explicitly ALLOWED for Banglish/URLs/loanwords
- Bengali vs Assamese script overlap: since Assamese docs are filtered at pipeline level, Assamese vocabulary patterns won't reach BPE merge frequency. The two unique Assamese characters get explicitly blocked.
- Zero tolerance: any Devanagari token in final vocab = fail, retrain

**Validation:**
- Tokenize 1000 random docs, check tokens-per-word ratio (target: 1.3-1.8)
- Verify common words get single tokens: "বাংলাদেশ", "সরকার", "তিনি"
- Verify Banglish tokenizes reasonably via byte fallback
- Compare efficiency against LLaMA/GPT tokenizers (expect 2-3x better)

**Critical checkpoint after tokenizer training:**
- Tokenize the ENTIRE cleaned corpus and count total tokens
- This number determines model size and epochs. Expected: ~600M-1B tokens from 5-6GB Bengali
- If < 1B tokens: consider adding Sangraha (AI4Bharat, 30B Bengali tokens) or multi-epoch training

**Output:** `bangla_32k.model` + `bangla_32k.vocab`

---

## Phase 3: Pre-training (300M)

**Architecture: LLaMA-style transformer (~306M params)**

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Layers | 18 | Gives ~306M params with d=1024 (24L would be ~398M) |
| Hidden dim | 1024 | Balances width vs depth, power-of-2 |
| Attention heads | 16 | 64 dim per head |
| KV heads | 4 | GQA - 4x less KV cache at inference |
| Intermediate (FFN) | 4096 | ~4x hidden dim, SwiGLU activation (3 projections) |
| Context length | 2048 | Sufficient for most Bangla docs, extendable later via RoPE scaling |
| Vocab size | 32,000 | From tokenizer |
| Position encoding | RoPE | Rotary embeddings |
| Normalization | RMSNorm | Faster than LayerNorm, same quality |
| Weight tying | Yes | Embedding weights shared with output head |
| Total params | ~306M | 18L * 15.2M/layer + 32.8M embeddings |

**Training setup:**
- Framework: HuggingFace Transformers + Accelerate
- Hardware: 1-2x A100 80GB (RunPod/Lambda/Vast.ai)
- Precision: BF16 mixed precision
- Tokens: determined after tokenizer training (estimate ~600M-1B from current corpus). Multi-epoch (3-4 epochs) to reach ~2-3B effective tokens. Add Sangraha for more data if needed.
- Effective batch size: ~512K tokens (micro-batch 8, gradient accumulation)
- Optimizer: AdamW, lr 3e-4, cosine decay to 3e-5, 2000 step warmup
- Training time: ~6-12 hours on 1x A100 (fewer layers = faster than 24L)

**Checkpointing:**
- Save every 1000 steps to Google Drive
- Keep last 5 checkpoints + best by validation loss
- Validation: held-out 1% of corpus, eval every 500 steps

---

## Phase 4: Fine-tuning

### Stage 1: Supervised Fine-Tuning (SFT)

Turn base model into an instruction-follower.

**Data:**
- Translate subset of OpenOrca or Alpaca to Bangla via GPT-4o/Claude (5K-10K high-quality pairs)
- Manually curate 500-1000 Bangla-native prompts: Q&A, summarization, creative writing, Bangladesh facts, Bangla grammar
- Use existing open Bangla instruction datasets from HuggingFace (verify quality first)

**Format:** ChatML-style with reserved special tokens: `<|user|>`, `<|assistant|>`, `<|end|>`

**Training:** 2-3 epochs, lr 2e-5, ~1-2 hours on A100

### Stage 2: DPO (Direct Preference Optimization)

Alignment without reward model complexity.

**Data:** Generate 2 responses per prompt from SFT model, have humans or stronger model pick the better one. 1K-3K preference pairs.

**Why DPO over RLHF:** At 300M scale, DPO gives 90% of alignment benefit with 10% of complexity. No reward model, no PPO instability.

**Training:** 1 epoch, lr 5e-6, beta=0.1, ~30 minutes on A100

### Stage 3 (optional): Task-specific LoRA adapters

For NLP foundation model goal:
- Sentiment analysis (Bangla product reviews)
- NER (Bangla named entities)
- Summarization (news articles from corpus)

Cheap LoRA fine-tunes, ~$2-5 each.

---

## Phase 5: Evaluation

### Layer 1: Perplexity and loss
- Held-out test set perplexity, tracked across training
- Compare against existing Bangla models where possible

### Layer 2: Bangla-specific benchmarks
- IndicNLPSuite: sentiment, NER, NLI (zero-shot and few-shot) - well-established, reproducible
- BanglaRQA: reading comprehension / QA (ACL Findings 2022)
- BanglaMATH: math reasoning at grade 6-8 level
- MBPP-Bangla: code generation with Bangla prompts (if applicable)
- Translation quality: English->Bangla BLEU/chrF++
- Bangla MMLU equivalent: create 200 multiple-choice questions (Bangla literature, Bangladesh history, general knowledge, science in Bangla) - community contribution
- Toxicity/safety: 500 completions from sensitive prompts

### Layer 3: Human evaluation
- 100 diverse prompts, 3 Bangla-native annotators
- Rate: fluency, factual accuracy, coherence, Bangla naturalness
- Compare against Google Translate-style output and existing Bangla models
- Contamination spot-check: annotators flag any Hindi/Assamese words in output. Zero tolerance.

### Layer 4: Tokenizer efficiency
- Tokens-per-sentence on held-out text vs multilingual tokenizers (LLaMA, GPT-4)
- Target: 2-3x more efficient

---

## Phase 6: Serving and Deployment

### Path A: API serving
- Quantize: GPTQ or AWQ to 4-bit (~200MB)
- Serve: vLLM with PagedAttention + continuous batching
- Hardware: single T4/L4 (~$0.20/hr) or RTX 3060
- API: FastAPI wrapper, OpenAI-compatible chat completions endpoint

### Path B: Edge/local deployment
- Convert to GGUF for llama.cpp
- Quantization: Q4_K_M (balanced), Q5_K_M (better quality), Q8_0 (near-lossless)
- 300M Q4_K_M runs on phones - on-device Bangla autocomplete is feasible

### Release strategy
- HuggingFace Hub: base model, SFT model, DPO model, GGUF quants, LoRA adapters (separate repos, one org)
- Model card with training details, data sources, benchmarks, limitations
- Tokenizer released alongside
- License: Apache 2.0 or community license (decide later)

---

## Data Scaling Strategy

Current corpus (~5-6GB) yields ~600M-1B tokens. Options to increase:
1. Add Sangraha (AI4Bharat) - claims 30B Bengali tokens, would be the single biggest source
2. Multi-epoch training - up to 4 epochs degrades gracefully (Muennighoff et al.)
3. Add more web crawl sources, Common Crawl direct filtering
4. Synthetic data augmentation in later phases

## Scale-up Path

After 306M is validated:
1. Measure actual token count, gather more data if needed (Sangraha priority)
2. Train 1B model with same pipeline (~$300-800)
3. Re-run full eval, compare against 306M
4. Repeat: more data -> larger model -> better results
