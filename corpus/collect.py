#!/usr/bin/env python3
"""Kotha corpus collector — resumable, metadata-tagging, GDrive-shipping.

Streams an HF source, routes each doc via script_purity.classify() into
Track A (native) / Track B (banglish) / reject, tags dialect + register +
provenance, writes gzipped JSONL shards, and (optionally) rclone-moves each
finished shard to Google Drive then deletes it locally — so a laptop with
little free disk can collect for weeks. Rejects that carry a same-script
contamination signal (Assamese) or a blocked script are sampled into an audit
stream: that stream is the raw material for the paper's contamination audit.

Designed to survive sleep / network drops / reboots: doc-count checkpoint per
source (resume via IterableDataset.skip), retryable, idempotent, tmux-friendly.

Examples:
  python3 corpus/collect.py --source demo                    # offline self-demo
  nice -n19 ionice -c3 python3 corpus/collect.py \\
      --source wikipedia_bn --shard-size 20000 \\
      --rclone-remote gdrive:kotha-corpus                    # real run
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_purity import classify          # noqa: E402
from dialect import tag as dialect_tag       # noqa: E402
import sources as source_registry            # noqa: E402


def script_of(v) -> str:
    if v.bengali_ratio >= 0.90:
        return "bengali"
    if v.latin_ratio >= 0.90:
        return "latin"
    return "mixed"


DEMO_DOCS = [
    ("বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র। এর রাজধানী ঢাকা এবং রাষ্ট্রভাষা বাংলা। "
     "মুক্তিযুদ্ধের মধ্য দিয়ে ১৯৭১ সালে দেশটি স্বাধীনতা অর্জন করে। বাংলা ভাষা এই জাতির "
     "সংস্কৃতির প্রাণকেন্দ্র এবং কোটি মানুষের মাতৃভাষা। ") * 3,          # native
    "tomar ki khobor? ami valo achi. tui kemon achis bhai, onek din por kotha holo",   # banglish
    "Bro ajke meeting ta cancel hoye gese, assignment ta submit korso? login korte partesi na",  # romanized banglish
    "আজকে office এ গিয়ে দেখি সব meeting cancel, assignment টা submit করতে হবে আজ রাতের মধ্যে ভাই।",  # bengali-script codeswitch
    "মুই ভাত খাম বাহে, তোই কি করিস? হামরা আইজ বাজারত যাম। এইঠে অনেক মানুষ আছে।",       # rangpur native
    "আঁই খিতা করমু, আমরার বাড়িত আইও। সিলেটি ভাষা খুব সুন্দর একটা ভাষা।",                # sylheti
    "बांग्ला भाषा सुंदर है और यह हिंदी वाक्य दूषण है यहाँ देवनागरी लिपि मिलेছে।",          # hindi contamination
    "মোৰ নাম ৰাম, মই ভাল আছোঁ। এইটো এটা অসমীয়া বাক্য যত ৰ আৰু ৱ আখৰ আছে বহুতো।",        # assamese
    "The weather is nice today and we are going out for a walk with our friends in the park.",  # english
]


class ShardWriter:
    def __init__(self, track: str, out_dir: Path, source: str, shard_size: int, remote: str | None):
        self.track, self.source, self.shard_size, self.remote = track, source, shard_size, remote
        self.dir = out_dir / track
        self.dir.mkdir(parents=True, exist_ok=True)
        self.idx = self._next_idx()
        self.n = 0
        self.fh = None

    def _next_idx(self) -> int:
        # Robust across restarts. Shards are rclone-*moved* to the remote and
        # deleted locally, so the old `len(local files)` restarted numbering low
        # and overwrote shards already shipped to the remote. Instead, continue
        # above the max shard index seen on BOTH local disk and the remote so
        # names never collide. Falls back to local-only (loudly) if rclone fails.
        import re
        pat = re.compile(
            rf"{re.escape(self.source)}-{re.escape(self.track)}-(\d+)\.jsonl\.gz$"
        )
        max_idx = -1
        for p in self.dir.glob(f"{self.source}-{self.track}-*.jsonl.gz"):
            m = pat.search(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if self.remote:
            try:
                r = subprocess.run(
                    ["rclone", "lsf", f"{self.remote}/{self.track}/"],
                    capture_output=True, text=True, timeout=180,
                )
                if r.returncode == 0:
                    for line in r.stdout.splitlines():
                        m = pat.search(line.strip())
                        if m:
                            max_idx = max(max_idx, int(m.group(1)))
                else:
                    print(f"  [idx WARN] rclone lsf failed for {self.track} "
                          f"(rc={r.returncode}); local-only max {max_idx}",
                          file=sys.stderr)
            except Exception as e:
                print(f"  [idx WARN] rclone lsf error for {self.track}: {e}; "
                      f"local-only max {max_idx}", file=sys.stderr)
        return max_idx + 1

    def _open(self):
        self.path = self.dir / f"{self.source}-{self.track}-{self.idx:05d}.jsonl.gz"
        self.fh = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, rec: dict):
        if self.fh is None:
            self._open()
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1
        if self.n >= self.shard_size:
            self._rotate()

    def _rotate(self):
        self.fh.close()
        self._ship(self.path)
        self.idx += 1
        self.n = 0
        self.fh = None

    def _ship(self, path: Path):
        if not self.remote:
            return
        dest = f"{self.remote}/{self.track}/"
        r = subprocess.run(["rclone", "move", str(path), dest], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [rclone WARN] {path.name}: {r.stderr.strip()[:120]}", file=sys.stderr)

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self._ship(self.path)
            self.fh = None


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def iter_source(name: str, limit: int, skip: int):
    if name == "demo":
        docs = DEMO_DOCS * (max(1, limit // len(DEMO_DOCS)) if limit else 1)
        for i, t in enumerate(docs[skip:]):
            yield t
        return
    meta = source_registry.get(name)
    field = meta["text_field"]
    local = meta.get("local_jsonl")
    if local:                                  # local parsed JSONL (e.g. wikisource dump)
        with open(local, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < skip:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line).get(field, "") or ""
                except json.JSONDecodeError:
                    continue
        return
    from datasets import load_dataset
    ds = load_dataset(meta["hf_id"], meta.get("config"), data_dir=meta.get("data_dir"),
                      split=meta["split"], streaming=True)
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        yield row.get(field, "") or ""


def main():
    ap = argparse.ArgumentParser(description="Kotha corpus collector")
    ap.add_argument("--source", required=True, help="registry key or 'demo'")
    ap.add_argument("--out", default="corpus/data", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = unbounded")
    ap.add_argument("--shard-size", type=int, default=50000)
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--rclone-remote", default=None, help="e.g. gdrive:kotha-corpus")
    ap.add_argument("--reject-sample", type=float, default=0.02, help="fraction of non-contamination rejects kept for audit")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--max-seconds", type=int, default=0, help="stop cleanly after N seconds (0 = unbounded)")
    args = ap.parse_args()

    rng = random.Random(0)
    state_path = args.state or (args.out / f"_state_{args.source}.json")
    state = load_state(state_path)
    seen = state.get(args.source, {}).get("seen", 0)
    src_meta = {} if args.source == "demo" else source_registry.get(args.source)

    writers = {t: ShardWriter(t, args.out, args.source, args.shard_size, args.rclone_remote)
               for t in ("native", "banglish", "codeswitch", "audit")}
    stats = Counter()
    dia_stats = Counter()
    kept_chars = 0
    samples = {"native": [], "banglish": [], "codeswitch": [], "audit": []}
    t0 = time.time()

    print(f"[collect] source={args.source} resume@{seen} out={args.out} "
          f"remote={args.rclone_remote or '(local)'}")
    try:
        for i, text in enumerate(iter_source(args.source, args.limit, seen)):
            if args.limit and i >= args.limit:
                break
            if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                print(f"[collect] reached --max-seconds ({args.max_seconds}s)")
                break
            v = classify(text)
            force = src_meta.get("force_track")   # trusted source (e.g. curated Banglish)
            track = force or v.track
            stats[f"route:{track}"] += 1
            # Audit stream (non-forced sources only) = Assamese candidates (kept OR
            # rejected — for human gold review) + other-script contamination + reject sample.
            if not force and (v.assamese_chars >= 3
                    or v.reason.startswith(("blocked_script", "foreign_script"))
                    or (v.track == "reject" and rng.random() < args.reject_sample)):
                arec = {"text": text[:4000], "source": args.source,
                        "license": src_meta.get("license", "demo"),
                        "route": v.track, "reason": v.reason,
                        "bengali_ratio": round(v.bengali_ratio, 3),
                        "assamese_chars": v.assamese_chars, "blocked_chars": v.blocked_chars}
                writers["audit"].write(arec)
                if len(samples["audit"]) < 3:
                    samples["audit"].append(arec)
            if not force and v.track == "reject":
                stats[f"reject:{v.reason.split('(')[0].split(':')[0]}"] += 1
                continue
            if force and len(text.strip()) < 20:
                stats["reject:empty"] += 1
                continue
            dia, dscore = dialect_tag(text)
            dia_stats[dia] += 1
            rec = {
                "text": text,
                "source": args.source,
                "license": src_meta.get("license", "demo"),
                "register": track,                        # native | banglish | codeswitch
                "dialect": dia,
                "script": script_of(v),
                "meta": {"bengali_ratio": round(v.bengali_ratio, 3),
                         "latin_ratio": round(v.latin_ratio, 3),
                         "assamese_chars": v.assamese_chars,
                         "dialect_score": dscore},
            }
            writers[track].write(rec)
            kept_chars += len(text)
            if len(samples[track]) < 3:
                samples[track].append(rec)

            if (i + 1) % args.checkpoint_every == 0:
                state.setdefault(args.source, {})["seen"] = seen + i + 1
                save_state(state_path, state)
                el = time.time() - t0
                print(f"  [{el:6.0f}s] {i + 1} docs | native={stats['route:native']} "
                      f"banglish={stats['route:banglish']} reject={stats['route:reject']} "
                      f"| ~{kept_chars / 4.45 / 1e6:.2f}M tok kept", flush=True)
    except KeyboardInterrupt:
        print("\n[collect] interrupted — flushing + checkpointing")
    finally:
        for w in writers.values():
            w.close()
        processed = i + 1 if 'i' in dir() else 0
        state.setdefault(args.source, {})["seen"] = seen + processed
        save_state(state_path, state)

    dt = time.time() - t0
    print(f"\n[collect] done in {dt:.1f}s — processed {processed} docs")
    print(f"  kept ~{kept_chars / 1e6:.1f}M chars ≈ {kept_chars / 4.45 / 1e6:.2f}M tokens "
          f"(~{kept_chars / 4.45 / max(1, dt) / 1000:.1f}k tok/s)")
    print("  routes:", dict(sorted(stats.items())))
    print("  dialects (kept docs):", dict(dia_stats))
    for track in ("native", "banglish", "codeswitch", "audit"):
        if samples[track]:
            print(f"\n  --- sample {track} record ---")
            s = samples[track][0]
            print("   " + json.dumps({k: (v[:90] + "…" if isinstance(v, str) and len(v) > 90 else v)
                                      for k, v in s.items()}, ensure_ascii=False))

    # HF streaming leaves C-extension prefetch threads that can segfault during
    # normal interpreter finalization. All shards + state are already flushed
    # above, so hard-exit cleanly for a reliable exit code in long tmux runs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
