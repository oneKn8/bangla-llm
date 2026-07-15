#!/usr/bin/env python3
"""Post-hoc metadata enrichment pass over the collected Kotha corpus.

Runs OFF the collection hot path so it applies uniformly to all shards (including
those already on Drive) and can scale to a pod. Adds, per document:
  provenance   : organic | transliterated | web | encyclopedic   (per-source map)
  has_emoji    : bool                                            (regex)
  emoji_count  : int
  profanity    : {hits, severity, count}                         (lexicon)
  emotion      : label + score        (optional; model-backed, see EmotionTagger)

Cheap taggers (emoji/profanity/provenance) run on CPU at high throughput. Emotion is
opt-in (--emotion) because it loads a model; intended for the DEDUPED corpus / SFT
candidate subsets, not every raw pretraining token. Resumable: skips shards already
present in --out-dir.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from corpus.lexicons.profanity_bn import find_profanity
from corpus.quality import quality_tags

# --- provenance: per-source constant. Central so it is auditable & editable. ---
PROVENANCE = {
    "banglatlit_pt": "organic",          # verified: native-written social comments
    "banglish_kawsar_v3": "transliterated",  # kawsar family = transliteration-derived (heuristic)
    "banglish_kawsar_v1": "transliterated",
    "banglish_kawsar_80k": "transliterated", # confirmed: Bengali<->Banglish<->English parallel
    "banglish_tensorlab_225k": "transliterated",
    "wikipedia_bn": "encyclopedic",
    "fineweb2_bn": "web", "hplt_bn": "web", "culturax_bn": "web", "sangraha_bn": "web",
}

# --- emoji: core pictographic ranges (deliberately excludes plain arrows/dingbats
#     that co-occur in normal text). ---
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF"   # symbols & pictographs, supplemental, ext-A
    "\U0001F1E6-\U0001F1FF"    # regional indicators (flags)
    "\U00002600-\U000026FF"    # miscellaneous symbols
    "\U00002700-\U000027BF"    # dingbats
    "\U0001F000-\U0001F0FF"    # mahjong/dominoes/cards
    "\U00002B00-\U00002BFF]"   # misc symbols & arrows (stars, etc.)
)


def emoji_stats(text: str) -> dict:
    ms = EMOJI_RE.findall(text)
    return {"has_emoji": bool(ms), "emoji_count": len(ms)}


class EmotionTagger:
    """Pluggable emotion classifier. Native (Bengali-script) and Banglish (romanized)
    may use different models per the research findings; wire concrete model ids here.
    Kept lazy so importing enrich.py stays cheap and CPU-only paths never load torch."""
    def __init__(self, native_model: str, banglish_model: str | None = None,
                 banglish_strategy: str = "native_model", batch_size: int = 32, device: int = -1):
        self.native_model, self.banglish_model = native_model, banglish_model
        self.banglish_strategy, self.batch_size, self.device = banglish_strategy, batch_size, device
        self._native = self._bl = None

    def _pipe(self, model):
        from transformers import pipeline
        return pipeline("text-classification", model=model, device=self.device,
                        truncation=True, max_length=256)

    def label(self, texts: list[str], registers: list[str]) -> list[dict]:
        # route by register; batch each group; None emotion if a route is disabled
        out = [None] * len(texts)
        native_idx = [i for i, r in enumerate(registers) if r == "native"]
        other_idx = [i for i in range(len(texts)) if i not in native_idx]
        if native_idx:
            if self._native is None:
                self._native = self._pipe(self.native_model)
            for i, pred in zip(native_idx, self._native([texts[i][:1000] for i in native_idx],
                                                        batch_size=self.batch_size)):
                out[i] = {"label": pred["label"], "score": round(pred["score"], 4)}
        if other_idx:
            if self.banglish_strategy == "skip":
                for i in other_idx:
                    out[i] = {"label": "unknown", "score": 0.0}
            else:
                mdl = self.banglish_model or self.native_model
                if self._bl is None:
                    self._bl = self._native if mdl == self.native_model and self._native else self._pipe(mdl)
                for i, pred in zip(other_idx, self._bl([texts[i][:1000] for i in other_idx],
                                                       batch_size=self.batch_size)):
                    out[i] = {"label": pred["label"], "score": round(pred["score"], 4)}
        return out


def enrich_doc(d: dict) -> dict:
    text = d.get("text", "")
    d["provenance"] = PROVENANCE.get(d.get("source", ""), "unknown")
    d.update(emoji_stats(text))
    d["profanity"] = find_profanity(text)
    d["quality"] = quality_tags(text)
    return d


def process_shard(in_path: str, out_path: str, emotion: EmotionTagger | None):
    rows = []
    try:
        with gzip.open(in_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(enrich_doc(json.loads(line)))
                except Exception:
                    continue
    except (EOFError, OSError):
        # truncated / mid-write shard (a live collector is still writing it):
        # skip entirely, do NOT write partial output (resumable skip would lock it in)
        return None
    if emotion and rows:
        preds = emotion.label([r.get("text", "") for r in rows],
                              [r.get("register", "") for r in rows])
        for r, p in zip(rows, preds):
            r["emotion"] = p
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="corpus/data", help="root with native/ banglish/ codeswitch/ subdirs")
    ap.add_argument("--out-dir", default="corpus/data_enriched")
    ap.add_argument("--emotion", action="store_true", help="enable model-based emotion tagging")
    ap.add_argument("--native-model", default="")
    ap.add_argument("--banglish-model", default="")
    ap.add_argument("--banglish-strategy", default="native_model", choices=["native_model", "skip"])
    ap.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 first GPU")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--min-age-seconds", type=int, default=120,
                    help="skip shards modified more recently than this (likely mid-write by a live collector)")
    args = ap.parse_args()

    emotion = None
    if args.emotion:
        if not args.native_model:
            sys.exit("--emotion requires --native-model")
        emotion = EmotionTagger(args.native_model, args.banglish_model or None,
                                args.banglish_strategy, device=args.device)

    shards = sorted(glob.glob(os.path.join(args.in_dir, "*", "*.jsonl.gz")))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    done = tot = 0
    for sp in shards:
        rel = os.path.relpath(sp, args.in_dir)
        op = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        if os.path.exists(op):
            done += 1
            continue
        if args.min_age_seconds and (time.time() - os.path.getmtime(sp)) < args.min_age_seconds:
            print(f"[enrich] skip recent (mid-write?): {rel}", flush=True)
            continue
        n = process_shard(sp, op, emotion)
        if n is None:
            print(f"[enrich] skip truncated: {rel}", flush=True)
            continue
        tot += n
        print(f"[enrich] {rel}: {n} docs", flush=True)
    print(f"[enrich] done: {len(shards)-done} shards processed, {done} skipped, {tot} docs enriched", flush=True)


if __name__ == "__main__":
    main()
