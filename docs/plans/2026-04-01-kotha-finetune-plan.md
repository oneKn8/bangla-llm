# Kotha-1: Fine-Tuning Pipeline Plan

**Model:** Kotha-1 (কথা-১) -- 306M parameter Bengali language model
**Goal:** Produce a publishable, instruction-following Bengali LLM for arXiv paper, HuggingFace release, and O-1A evidence.
**Base checkpoint:** training/checkpoints/step-4000 (loss 5.45, 2B effective tokens)

---

## Phase 1: Data Curation (2 hrs, local CPU)

**Input:** 342K pairs from Bangla-Instruct + 5K templates + Gemini-generated pairs

**Filtering pipeline:**
1. Remove pairs where response < 50 chars
2. Remove pairs where Bengali char ratio < 50% in response
3. Remove exact duplicate instructions
4. Remove near-duplicate instructions (first 50 chars match)
5. Length cap: remove responses > 4096 chars (likely garbage)
6. Spot-check 100 random samples for quality

**Output:**
- finetune/data/kotha_sft_train.jsonl (95%)
- finetune/data/kotha_sft_val.jsonl (2.5%)
- finetune/data/kotha_sft_test.jsonl (2.5%)
- Target: 50-100K curated pairs

---

## Phase 2: Continued Pre-Training (2-3 hrs, A100)

**Purpose:** Improve base model Bengali fluency before SFT.

**Data:**
- 87K Bangla-TextBook passages (9.9M tokens)
- Response-only text from filtered SFT data (raw language modeling)

**Config:**
- Base: training/checkpoints/step-4000
- Objective: causal LM (same as pre-training)
- LR: 1e-5, cosine decay
- Epochs: 2
- Batch: 8 micro x 16 grad_accum = 128 effective
- Seq len: 2048
- num_workers: 0 (bug fix applied)
- Save: every epoch + final

**Output:** checkpoints/kotha-cpt-final/

---

## Phase 3: Full SFT (3-4 hrs, A100)

**Purpose:** Instruction-following capability. Full parameter update.

**Data:** Curated dataset from Phase 1

**Config:**
- Base: checkpoints/kotha-cpt-final/ (Phase 2 output)
- All 306M params updated (no LoRA)
- LR: 2e-5, cosine decay
- Warmup: 5% of total steps
- Epochs: 2
- Batch: 4 micro x 8 grad_accum = 32 effective
- Seq len: 2048
- Gradient checkpointing: enabled
- BF16: yes
- Format: ChatML (<|im_start|>, <|im_end|>)
- Save: every epoch + best by val loss

**Output:** checkpoints/kotha-1-sft/

---

## Phase 4: News-Domain LoRA (1 hr, A100)

**Purpose:** Domain-specific adapter for Dr. Khan's recommendation letter.

**Data:** Newspaper articles from data pipeline (Prothom Alo, Kaler Kantho)
- Summarization pairs
- Headline generation pairs
- News Q&A pairs
- Target: 5-10K news instruction pairs

**Config:**
- Base: checkpoints/kotha-1-sft/ (Phase 3 output)
- LoRA: r=16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LR: 2e-5
- Epochs: 3
- Batch: 4 micro x 4 grad_accum = 16 effective

**Output:** checkpoints/kotha-1-news-lora/

---

## Phase 5: Evaluation (4 hrs, local)

**Benchmarks:**

1. Perplexity (held-out test sets: Wikipedia, newspaper, textbook)
   - Report: base -> CPT -> SFT progression

2. Tokenizer efficiency
   - Metric: chars/token on Bengali text
   - Compare: Kotha-32K vs LLaMA tokenizer vs BLOOM tokenizer vs GPT tokenizer

3. Generation quality (human eval)
   - 100 random prompts, rate fluency/accuracy/coherence 1-5
   - Compare: Kotha-1 vs BLOOM-560M vs mGPT

4. Instruction following
   - 50 diverse Bengali instructions, blind comparison

5. News domain (Phase 4)
   - ROUGE scores on held-out news summaries
   - Base SFT vs news LoRA

---

## Phase 6: Release

**HuggingFace (shifat-santo/):**
- kotha-1-base (pre-trained)
- kotha-1 (SFT, instruction-following)
- kotha-1-news (LoRA adapter)
- kotha-tokenizer (32K Bengali BPE)
- kotha-sft-data (curated dataset)

**arXiv paper:**
- Title: "Kotha: A 306M Parameter Bengali Language Model Trained from Scratch"
- 5 contributions: data pipeline, tokenizer, base model, instruction tuning, domain adaptation

---

## Compute Budget

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| 1. Data curation | Local CPU | 2 hrs | $0 |
| 2. Continued pre-training | A100 80GB | 2-3 hrs | ~$4 |
| 3. Full SFT | A100 80GB | 3-4 hrs | ~$6 |
| 4. News LoRA | A100 80GB | 1 hr | ~$2 |
| 5. Evaluation | Local CPU/GPU | 4 hrs | $0 |
| 6. Release | N/A | 1-2 days | $0 |
| **Total** | | **~8 hrs A100** | **~$12** |

---

## Execution Order

Phases 2-4 run sequentially on a single A100 session (~7 hrs).
Phase 1 must complete before spinning up the instance.
Phase 5 runs locally after downloading all checkpoints.
Phase 6 after evaluation is complete.
