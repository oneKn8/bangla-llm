#!/usr/bin/env python3
"""Kotha corpus — MinHash-LSH NEAR-deduplication (final released-split stage).

Second and final dedup stage, run AFTER corpus/process_corpus.py has already done
exact cross-source dedup. This one removes NEAR-duplicates (Jaccard ~>= 0.8 on word
5-grams) and writes the physical released split, routed into the three redistribution
buckets from corpus/release_split.py (releasable / conditional / train_only).

Pipeline (two-phase MinHash-LSH):
  1. Shingle each doc: NFC-normalize -> whitespace words -> word 5-grams -> xxh64 int.
  2. MinHash signature (128 perms), numpy-vectorized universal hash over a Mersenne
     prime P = 2**61-1. The signature array is a single uint32 [N,128] block.
  3. LSH banding (b=16, r=8; collision threshold ~ (1/b)**(1/r) = 2**-0.5 ~ 0.71): docs sharing
     any band bucket are candidate near-dupes.
  4. Union-find (rank + vectorized pointer-doubling) over within-bucket chains -> clusters.
  5. Keep one representative per cluster (highest SOURCE_PRIORITY, tie-break longest text);
     singletons always kept.
  6. Write each kept doc to <out>/<bucket>/<bucket>-<NNNNN>.jsonl.gz (rotating shards).
  7. Exact SentencePiece token count per kept doc, accumulated per bucket, plus a
     JSON report of docs_in / kept / removed / clusters / per-bucket tokens+sadhu.

CORRECTNESS NOTE (why the modmul is not the naive one-liner): the universal hash
  h_i(x) = ((a_i*x + b_i) mod P) mod 2**32
with a 64-bit shingle x and a_i up to P ~ 2**61 has products up to ~2**125, which
silently WRAP in numpy uint64 (this is what datasketch / text-dedup do and it is NOT
the stated formula). We instead reduce x mod P and run an exact split-limb modular
multiply whose every intermediate stays < 2**64. `--self-test` proves this vectorized
hash equals the arbitrary-precision Python formula on random 64-bit inputs.

PARALLELISM (SHARD-LEVEL): two shard-parallel reads of --shards-dir. Global doc order ==
concatenation of shards in the fixed iter_shards priority order (docs in file order).
Pass 1 pools over shards -> each worker signs a whole shard and returns one compact block;
the parent reassembles blocks in shard order at cumulative offsets (imap preserves order),
producing global arrays identical to a single reader. Clustering is single-threaded (fast).
Pass 2 pools over shards again -> each worker re-reads its shard, batch-tokenizes kept docs,
and writes its own bucket files. `--workers 1` runs both passes serially (no pool) and gives
identical results at the doc-set level. The MinHash algorithm and exact modmul are unchanged.

RESOURCE ENVELOPE (target: the ~24.5-27.8M-doc exact-deduped Kotha corpus):
  RAM  : signatures uint32[N,128] ~= 12.5-14 GB at 25-28M docs; the pass-1 reassembly
         concatenate copies once (~2x sig transient) + np.unique band transients (~2-3 GB)
         + union-find (~0.2 GB). Peak ~= 28-32 GB. Recommend a 48-64 GB pod. Workers hold
         only one shard each. `--sample N` caps N for local testing.
  TIME : near-linear in --workers for both passes until I/O-bound. On 16-32 cores expect
         the signature pass to drop from ~15-36 h (1 core) to roughly 1-3 h. One-time job.

Examples:
  # correctness check (no shards, no output) -- run this first
  python3 corpus/near_dedup.py --self-test
  # full run over the local mirror
  python3 corpus/near_dedup.py --shards-dir /home/oneknight/kotha-mirror \\
      --tracks native codeswitch banglish \\
      --out corpus/released --report corpus/near_dedup_report.json
  # quick local smoke test on a small sample
  python3 corpus/near_dedup.py --shards-dir /home/oneknight/kotha-mirror \\
      --sample 50000 --out /tmp/nd_out --report /tmp/nd_report.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

# Pin math libraries to ONE thread per process before numpy is imported. Each worker
# handles one shard on one core; without this every worker would spin up its own BLAS/
# OpenMP thread pool (workers x threads = oversubscription that wrecks parallel scaling,
# worst on many-core pods). fork inherits these into the workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xxhash                                             # noqa: E402
from process_corpus import (                             # noqa: E402
    SOURCE_PRIORITY, shard_source, iter_shards, _safe_lines,
)
from release_split import classify as release_classify   # noqa: E402
from sadhu import classify as sadhu_classify             # noqa: E402

# --- Mersenne-prime modular-arithmetic constants (all np.uint64) ------------------
# P = 2**61-1 is prime and > 2**32; 2**61 == 1 (mod P) gives cheap fold reduction.
_P = np.uint64((1 << 61) - 1)
_MASK32 = np.uint64((1 << 32) - 1)
_M31 = np.uint64((1 << 31) - 1)
_M30 = np.uint64((1 << 30) - 1)
_S61 = np.uint64(61)
_S31 = np.uint64(31)
_S30 = np.uint64(30)
_TWO = np.uint64(2)
_P_PY = (1 << 61) - 1   # python-int copy for the reference check


# ------------------------- exact modular arithmetic (vectorized) ------------------
def _mod_p(v: np.ndarray) -> np.ndarray:
    """v mod P for any uint64 array v < 2**64, via two Mersenne folds + one subtract.
    Because 2**61 == 1 (mod P): v == (v & P) + (v >> 61) (mod P). Two folds bring any
    64-bit value below 2**61, then one conditional subtract yields a value in [0, P)."""
    v = (v & _P) + (v >> _S61)
    v = (v & _P) + (v >> _S61)
    return np.where(v >= _P, v - _P, v)


def _mulmod(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(a * x) mod P for uint64 arrays with a < P and x < P (both < 2**61), broadcast.
    Split-limb schoolbook multiply where EVERY intermediate stays < 2**64 (no wrap):
      a = aH*2**31 + aL, x = xH*2**31 + xL  (aL,xL < 2**31; aH,xH < 2**30)
      a*x = aH*xH*2**62 + (aH*xL + aL*xH)*2**31 + aL*xL
    and each coefficient of 2**k is reduced with 2**61 == 1 (mod P) before it can grow."""
    aL = a & _M31
    aH = a >> _S31
    xL = x & _M31
    xH = x >> _S31
    t0 = _mod_p(aL * xL)                     # constant term ( < 2**62 )
    t2 = _mod_p((aH * xH) * _TWO)            # 2**62 term -> *2 (mod P); aH*xH < 2**60
    mid = _mod_p(aH * xL + aL * xH)          # 2**31 term, reduced ( < 2**62 -> < P )
    # multiply the reduced middle term by 2**31 (mod P) without overflow:
    # mid < 2**61, so (mid<<31) mod P == (mid>>30) + ((mid & (2**30-1))<<31), then reduce.
    tmid = _mod_p((mid >> _S30) + ((mid & _M30) << _S31))
    return _mod_p(t0 + t2 + tmid)            # sum < 3P < 2**63


def _perm_hash(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Universal hash h(x) = ((a*x + b) mod P) mod 2**32, exact, uint64 in/out.
    x must already be reduced mod P (see _mod_p); a, b in [0, P)."""
    return _mod_p(_mulmod(a, x) + b) & _MASK32


def build_permutations(n_perm: int, seed: int):
    """128 independent (a, b) pairs for the universal hash family, from a fixed seed.
    a in [1, P) (nonzero multiplier), b in [0, P). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P_PY, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, _P_PY, size=n_perm, dtype=np.uint64)
    return a, b


# ------------------------------- shingling + signatures ---------------------------
def shingle_reduced(text: str, ngram: int) -> np.ndarray:
    """NFC-normalize, whitespace-tokenize, form word n-grams, xxh64 each to a 64-bit int,
    de-duplicate, and reduce mod P. Never returns empty: a doc with < ngram words uses
    its whole word list as a single shingle (an empty/whitespace doc -> the '' shingle)."""
    nfc = unicodedata.normalize("NFC", text)
    words = nfc.split()
    if len(words) < ngram:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
    h = np.fromiter(
        (xxhash.xxh64(g.encode("utf-8")).intdigest() for g in grams),
        dtype=np.uint64, count=len(grams),
    )
    h = np.unique(h)                 # MinHash is over the SET of shingles
    return _mod_p(h)


def doc_signature(text: str, perms, ngram: int, chunk: int = 4096) -> np.ndarray:
    """MinHash signature (uint32[n_perm]) = per-permutation min over the doc's shingles.
    Shingles are processed in chunks so the transient [chunk, n_perm] hash block is
    bounded regardless of document length (running elementwise min across chunks)."""
    a, b = perms
    n_perm = a.shape[0]
    xr = shingle_reduced(text, ngram)
    a_row = a.reshape(1, -1)
    b_row = b.reshape(1, -1)
    best = np.full(n_perm, np.uint64(1 << 32), dtype=np.uint64)   # > any 32-bit hash
    for s in range(0, xr.shape[0], chunk):
        col = xr[s:s + chunk].reshape(-1, 1)                      # [c, 1]
        block = _perm_hash(col, a_row, b_row)                     # [c, n_perm], < 2**32
        best = np.minimum(best, block.min(axis=0))
    return best.astype(np.uint32)


# ------------------ per-doc work unit (shared by serial + parallel paths) ---------
def _signature_row(text: str, perms, ngram: int):
    """All CPU-bound per-doc work: (uint32 MinHash signature, NFC char length, is_sadhu).
    Called verbatim by the single-threaded path AND by the multiprocessing workers, so
    the numbers are byte-identical regardless of --workers. NFC is applied once here;
    doc_signature re-normalizes internally (NFC is idempotent), matching the serial path."""
    nfc = unicodedata.normalize("NFC", text)
    sig = doc_signature(nfc, perms, ngram)
    return sig, len(nfc), bool(sadhu_classify(nfc).is_sadhu)


# ---------------------- shard enumeration + reading (deterministic order) ---------
def enumerate_shards(shards_dir: str, tracks: list[str]):
    """Ordered list of shard descriptors (source, track, path, shard_name). The order is
    IDENTICAL to process_corpus.iter_shards (sort by descending SOURCE_PRIORITY, then
    source name, then shard name). This defines every doc's global index: the global doc
    order == concatenation of shards in this order, docs in file order. That is what makes
    the shard-parallel result identical (at the doc-set level) to a single reader."""
    entries = []  # (-priority, source, track, path, shard_name)
    for track in tracks:
        d = Path(shards_dir) / track
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl.gz")):
            src = shard_source(p.name)
            entries.append((-SOURCE_PRIORITY.get(src, 0), src, track, str(p), p.name))
    entries.sort(key=lambda e: (e[0], e[1], e[4]))     # -priority, source, shard_name
    return [(src, track, path, name) for (_pri, src, track, path, name) in entries]


def read_shard_docs(path: str, shard_name: str):
    """Yield valid doc dicts from one local shard in file order, with the SAME skip logic
    as iter_docs (strip, skip blank, skip bad-JSON) and the same truncation tolerance, so
    pass 2 and pass 3 see the exact same doc sequence for a shard."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in _safe_lines(fh, shard_name):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield d


# ---- pass-2 worker: one whole shard -> one compact signature block (one pickle/shard) ----
_P2: dict = {}


def _init_pass2(n_perm: int, seed: int, ngram: int) -> None:
    # Permutations rebuilt from the same (n_perm, seed) in every process, so
    # build_permutations' determinism guarantees identical (a, b) tables everywhere.
    _P2["perms"] = build_permutations(n_perm, seed)
    _P2["ngram"] = ngram


def _pass2_worker(job):
    """job = (shard_index, src_default, path, shard_name). Reads the shard, computes each
    doc's (signature, text_len, source, is_sadhu), returns ONE compact block:
    (shard_index, sig[k,128] uint32, text_len[k] int64, src[k] list, is_sadhu[k] bool, k).
    src_default = shard_source(name), matching iter_docs' `d.get('source') or shard_source`."""
    shard_index, src_default, path, shard_name = job
    perms = _P2["perms"]
    ngram = _P2["ngram"]
    n_perm = perms[0].shape[0]
    sigs, lens, srcs, sadhus = [], [], [], []
    for d in read_shard_docs(path, shard_name):
        text = d.get("text", "") or ""
        s, tl, sad = _signature_row(text, perms, ngram)
        sigs.append(s)
        lens.append(tl)
        srcs.append(d.get("source") or src_default)
        sadhus.append(sad)
    k = len(sigs)
    sig = np.stack(sigs).astype(np.uint32) if k else np.empty((0, n_perm), dtype=np.uint32)
    return (shard_index, sig, np.asarray(lens, dtype=np.int64), srcs,
            np.asarray(sadhus, dtype=bool), k)


# ---- pass-3 worker: one whole shard -> its own bucket files + per-bucket tallies ----
_P3: dict = {}


def _init_pass3(tokenizer_path: str) -> None:
    import sentencepiece as _spm
    _P3["sp"] = _spm.SentencePieceProcessor(model_file=tokenizer_path)


def _pass3_worker(job):
    """job = (shard_index, src_default, path, shard_name, out, k, keep[k], is_sadhu[k]).
    Re-reads the shard; for kept docs (in file order, first k) batch-tokenizes, routes by
    release bucket, and writes to its OWN files <out>/<bucket>/<bucket>-s{idx:05d}.jsonl.gz
    at gzip level 4. Returns {bucket: (docs, tokens, sadhu)} for this shard."""
    shard_index, src_default, path, shard_name, out, k, keep, sadhu = job
    sp = _P3["sp"]
    outdir = Path(out)
    writers: dict = {}
    tally: dict = {}                       # bucket -> [docs, tokens, sadhu]
    bd, bt, bs, bl = [], [], [], []        # batch: doc, nfc-text, src, local-index

    def flush():
        if not bt:
            return
        enc = sp.encode(bt)                # C++ batch encode
        for d, src, jloc, ids in zip(bd, bs, bl, enc):
            bucket = release_classify(src)["bucket"]
            w = writers.get(bucket)
            if w is None:
                w = writers[bucket] = gzip.open(
                    outdir / bucket / f"{bucket}-s{shard_index:05d}.jsonl.gz",
                    "wt", encoding="utf-8", compresslevel=4)
                tally[bucket] = [0, 0, 0]
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            t = tally[bucket]
            t[0] += 1
            t[1] += len(ids)
            t[2] += 1 if bool(sadhu[jloc]) else 0
        bd.clear()
        bt.clear()
        bs.clear()
        bl.clear()

    j = 0
    for d in read_shard_docs(path, shard_name):
        if j >= k:
            break
        if keep[j]:
            text = d.get("text", "") or ""
            bd.append(d)
            bt.append(unicodedata.normalize("NFC", text))
            bs.append(d.get("source") or src_default)
            bl.append(j)
            if len(bt) >= 1000:
                flush()
        j += 1
    flush()
    for w in writers.values():
        w.close()
    return {b: tuple(v) for b, v in tally.items()}


# ----------------------------- LSH banding + clustering ---------------------------
def lsh_cluster(sig: np.ndarray, bands: int, rows: int):
    """Band the [N, bands*rows] signature into `bands` slices of `rows` rows; docs whose
    full band slice is byte-identical share a bucket and are unioned. Returns the root
    (component id) array of length N. Buckets are found EXACTLY via np.unique(axis=0)
    (no band-hash collisions); within a bucket we chain-union (O(k), never O(k**2))."""
    n = sig.shape[0]
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for band in range(bands):
        sl = sig[:, band * rows:(band + 1) * rows]
        # group identical band rows; only groups of size > 1 can be near-dupes
        _uniq, inv, counts = np.unique(sl, axis=0, return_inverse=True,
                                       return_counts=True)
        inv = inv.reshape(-1)
        cand = np.nonzero(counts[inv] > 1)[0]
        if cand.size == 0:
            continue
        # order candidate docs by their group id, then chain-union consecutive members
        order = cand[np.argsort(inv[cand], kind="stable")]
        g = inv[order]
        for k in range(1, order.size):
            if g[k] == g[k - 1]:
                union(int(order[k]), int(order[k - 1]))

    # resolve every node to its component root by vectorized pointer-doubling
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return parent


def select_keep(root: np.ndarray, priority: np.ndarray, text_len: np.ndarray) -> np.ndarray:
    """One representative per component: highest priority, tie-break longest text, final
    tie-break smallest doc index (determinism). Returns a boolean keep mask of length N."""
    n = root.shape[0]
    idx = np.arange(n, dtype=np.int64)
    # lexsort primary key is the LAST: group by root, then priority desc, len desc, idx asc
    order = np.lexsort((idx,
                        -text_len.astype(np.int64),
                        -priority.astype(np.int64),
                        root))
    root_sorted = root[order]
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = root_sorted[1:] != root_sorted[:-1]
    keep = np.zeros(n, dtype=bool)
    keep[order[is_start]] = True
    return keep


def cluster_stats(root: np.ndarray) -> dict:
    """Component-size accounting: kept == #components, clusters == components with >1
    member (the near-dup groups), singletons == components with exactly 1 member."""
    _roots, counts = np.unique(root, return_counts=True)
    n = int(root.shape[0])
    kept = int(counts.shape[0])
    return {
        "docs_in": n,
        "docs_kept": kept,
        "near_dup_removed": n - kept,
        "near_dup_removed_pct": round(100.0 * (n - kept) / max(1, n), 4),
        "clusters": int((counts > 1).sum()),
        "singletons": int((counts == 1).sum()),
    }


def decide(sig: np.ndarray, src_ids: np.ndarray, text_len: np.ndarray,
           sources: list[str], bands: int, rows: int):
    """Full decision path shared by the real run and the self-test: signatures ->
    LSH clusters -> keep mask + stats. `sources[src_ids[i]]` is doc i's source name."""
    priority = np.array([SOURCE_PRIORITY.get(s, 0) for s in sources],
                        dtype=np.int64)[src_ids]
    root = lsh_cluster(sig, bands, rows)
    keep = select_keep(root, priority, text_len)
    return keep, cluster_stats(root)


# ---------------------------------- shard streaming -------------------------------
def iter_docs(shards_dir: str, tracks: list[str], sample: int = 0):
    """Yield (doc_dict, source) for every valid JSONL line under <shards-dir>/<track>/,
    in the deterministic priority order of process_corpus.iter_shards. Same generator is
    used for the count / fill / write passes so doc index i is identical across passes."""
    n = 0
    for source, _track, shard, fh in iter_shards(None, shards_dir, tracks):
        for line in _safe_lines(fh, shard):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = d.get("source") or shard_source(shard)
            yield d, src
            n += 1
            if sample and n >= sample:
                return


# --------------------------------------- self test --------------------------------
def _verify_modmul(n: int = 4000, seed: int = 7) -> None:
    """Prove the vectorized hash equals the arbitrary-precision Python formula
    ((a*x + b) mod P) mod 2**32 on random FULL 64-bit shingles x and full-range a,b.
    This is the load-bearing correctness check for the exact modmul (no uint64 wrap)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P_PY, size=n, dtype=np.uint64)
    b = rng.integers(0, _P_PY, size=n, dtype=np.uint64)
    hi = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    x = (hi << np.uint64(32)) | lo                    # full 64-bit shingle hashes
    got = _perm_hash(_mod_p(x), a, b)                 # reduce x mod P, then hash
    bad = 0
    for i in range(n):
        ref = ((int(a[i]) * int(x[i]) + int(b[i])) % _P_PY) % (1 << 32)
        if int(got[i]) != ref:
            bad += 1
    ok = bad == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] exact modmul == big-int ((a*x+b) mod P) mod 2^32"
          f"  ({n} random 64-bit cases, {bad} mismatch)")
    if not ok:
        raise AssertionError("modmul mismatch vs arbitrary-precision reference")


def _mixed_words(n: int, salt: str) -> list[str]:
    """Deterministic list of n distinct Bengali/ascii tokens (a synthetic paragraph)."""
    bn = ["আমি", "তুমি", "সে", "আমরা", "শহর", "নদী", "আকাশ", "বই", "পড়া", "লেখা",
          "সকাল", "বিকাল", "রাত", "মানুষ", "কথা", "গান", "ছবি", "পথ", "বাড়ি", "গাছ"]
    out = []
    for i in range(n):
        out.append(f"{bn[i % len(bn)]}{salt}{i}" if i % 2 == 0 else f"word{salt}{i}")
    return out


def run_self_test(bands: int, rows: int, ngram: int, seed: int) -> int:
    """Build 5 synthetic docs and run the REAL signature+LSH+cluster path over them.
    A ~ 60-word paragraph; B = A with 2 adjacent words changed (near-dup, Jaccard > 0.8);
    C = A with ~40% of words changed (not a near-dup); D = a different paragraph;
    E = an exact copy of A. Expect {A,B,E} -> one cluster, C and D each their own =>
    kept == 3, near_dup_removed == 2. Returns 0 on all-pass, non-zero otherwise."""
    print("=" * 72)
    print(f"SELF-TEST  bands={bands} rows={rows} n_perm={bands * rows} ngram={ngram} seed={seed}")
    _verify_modmul()

    wA = _mixed_words(60, "a")
    A = " ".join(wA)
    wB = list(wA)                    # 2 ADJACENT word changes -> Jaccard ~0.81 ( > 0.8 )
    wB[29] = "CHANGEDxx"
    wB[30] = "CHANGEDyy"
    B = " ".join(wB)
    wC = list(wA)                    # ~40% (24/60) words changed, spread out -> not near-dup
    for j in range(0, 60, 5):
        wC[j] = f"different{j}"
        if j + 1 < 60:
            wC[j + 1] = f"other{j}"
    C = " ".join(wC)
    D = " ".join(_mixed_words(60, "d"))     # unrelated paragraph
    E = A                                    # exact copy of A

    docs = [A, B, C, D, E]
    labels = ["A", "B", "C", "D", "E"]
    perms = build_permutations(bands * rows, seed)
    sig = np.stack([doc_signature(t, perms, ngram) for t in docs]).astype(np.uint32)
    src_ids = np.zeros(len(docs), dtype=np.int32)            # same source for all
    text_len = np.array([len(t) for t in docs], dtype=np.int64)
    keep, stats = decide(sig, src_ids, text_len, ["wikipedia_bn"], bands, rows)
    root = lsh_cluster(sig, bands, rows)

    r = {labels[i]: int(root[i]) for i in range(len(docs))}
    print(f"  roots: {r}")
    print(f"  kept flags: {{" + ", ".join(f'{labels[i]}:{bool(keep[i])}'
                                          for i in range(len(docs))) + "}}")
    print(f"  stats: kept={stats['docs_kept']} removed={stats['near_dup_removed']} "
          f"clusters={stats['clusters']} singletons={stats['singletons']}")

    checks = []
    checks.append(("A,B,E share one cluster", r["A"] == r["B"] == r["E"]))
    checks.append(("C is its own cluster",
                   r["C"] != r["A"] and r["C"] != r["D"]))
    checks.append(("D is its own cluster",
                   r["D"] != r["A"] and r["D"] != r["C"]))
    checks.append(("exactly one of {A,B,E} kept",
                   int(keep[0]) + int(keep[1]) + int(keep[4]) == 1))
    checks.append(("C kept", bool(keep[2])))
    checks.append(("D kept", bool(keep[3])))
    checks.append(("total kept == 3", stats["docs_kept"] == 3))
    checks.append(("near_dup_removed == 2", stats["near_dup_removed"] == 2))

    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("=" * 72)
    print("SELF-TEST RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


# ------------------------------------------ main ----------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Kotha MinHash-LSH near-dedup + released split")
    ap.add_argument("--shards-dir", help="dir containing <track>/*.jsonl.gz")
    ap.add_argument("--tracks", nargs="+", default=["native", "codeswitch", "banglish"])
    ap.add_argument("--out", help="output root; writes <out>/<bucket>/<bucket>-sNNNNN.jsonl.gz "
                                  "(one file per input shard per bucket)")
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--tokenizer", default="tokenizer/output/bangla_bpe_32k.model",
                    help="SentencePiece .model for EXACT token counts")
    ap.add_argument("--shard-size", type=int, default=50_000,
                    help="legacy/unused: output shards now mirror input shards")
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--sample", type=int, default=0, help="cap docs (0 = all); for local testing")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel processes over shards for both passes; 1 = serial path")
    ap.add_argument("--self-test", action="store_true", help="run correctness self-test and exit")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    if args.self_test:
        return run_self_test(args.bands, args.rows, args.ngram, args.seed)

    if not args.shards_dir or not args.out:
        ap.error("--shards-dir and --out are required (or use --self-test)")

    n_perm = args.bands * args.rows
    t0 = time.time()

    shards = enumerate_shards(args.shards_dir, args.tracks)
    if not shards:
        print("no shards found under --shards-dir -- nothing to do", file=sys.stderr)
        return 1

    # --- pass 1/2: MinHash signatures + text length + সাধু flag, PARALLEL over SHARDS ---
    # Each worker processes one whole shard and returns a compact block (one pickle/shard);
    # the parent reassembles blocks in shard order at cumulative offsets. imap preserves
    # order, so the reassembled global arrays are identical to a single reader's order.
    print(f"pass 1/2: MinHash signatures ({n_perm} perms) over {len(shards):,} shards, "
          f"workers={args.workers} ...", flush=True)
    t_p2 = time.time()

    blocks = []      # (sig[k,128], text_len[k], src[k] list, is_sadhu[k]) in shard order
    spans = []       # (shard_index, k, offset) for the write pass
    state = {"total": 0, "expected": 0}

    def _consume(block_iter) -> None:
        # Reassemble blocks in shard order. For --sample we include whole shards in order
        # and truncate the shard that crosses N to exactly N-offset docs, reproducing the
        # single-reader sample semantics (the first N docs in global order).
        for blk in block_iter:
            si, bsig, blen, bsrc, bsad, k = blk
            if si != state["expected"]:
                raise RuntimeError(f"shard order mismatch: got {si}, expected {state['expected']}")
            state["expected"] += 1
            if args.sample:
                room = args.sample - state["total"]
                if room <= 0:
                    return
                if k > room:
                    bsig, blen, bsrc, bsad, k = (bsig[:room], blen[:room], bsrc[:room],
                                                 bsad[:room], room)
            spans.append((si, k, state["total"]))
            blocks.append((bsig, blen, bsrc, bsad))
            state["total"] += k
            if (state["expected"]) % max(1, args.progress_every // 20000 or 1) == 0:
                print(f"  [{time.time() - t0:6.0f}s] shards {state['expected']:,}/{len(shards):,}"
                      f"  docs {state['total']:,}", flush=True)
            if args.sample and state["total"] >= args.sample:
                return

    if args.workers <= 1:
        _init_pass2(n_perm, args.seed, args.ngram)
        _consume(_pass2_worker((i, s, p, nm)) for i, (s, _tr, p, nm) in enumerate(shards))
    else:
        import multiprocessing as _mp
        jobs = ((i, s, p, nm) for i, (s, _tr, p, nm) in enumerate(shards))
        with _mp.Pool(args.workers, initializer=_init_pass2,
                      initargs=(n_perm, args.seed, args.ngram)) as pool:
            _consume(pool.imap(_pass2_worker, jobs, chunksize=1))

    n_docs = state["total"]
    if n_docs == 0:
        print("no docs found -- nothing to do", file=sys.stderr)
        return 1

    # Reassemble global arrays (shard order). concatenate copies once (~2x sig transient).
    sig = np.concatenate([b[0] for b in blocks]) if len(blocks) > 1 else blocks[0][0]
    text_len = np.concatenate([b[1] for b in blocks]) if len(blocks) > 1 else blocks[0][1]
    is_sadhu = np.concatenate([b[3] for b in blocks]) if len(blocks) > 1 else blocks[0][3]
    src_index: dict[str, int] = {}
    sources: list[str] = []
    src_ids = np.empty(n_docs, dtype=np.int32)
    pos = 0
    for _bsig, _blen, bsrc, _bsad in blocks:     # intern src in global doc order
        for s in bsrc:
            sid = src_index.get(s)
            if sid is None:
                sid = len(sources)
                src_index[s] = sid
                sources.append(s)
            src_ids[pos] = sid
            pos += 1
    del blocks
    print(f"  pass 1 done: {n_docs:,} docs in {time.time() - t_p2:.1f}s "
          f"({1000 * (time.time() - t_p2) / max(1, n_docs):.3f} ms/doc)", flush=True)

    # --- LSH banding + union-find clustering + representative selection (single-threaded) ---
    print("clustering (LSH banding + union-find) ...", flush=True)
    keep, stats = decide(sig, src_ids, text_len, sources, args.bands, args.rows)
    del sig, text_len, src_ids
    print(f"  docs_in={stats['docs_in']:,} kept={stats['docs_kept']:,} "
          f"removed={stats['near_dup_removed']:,} "
          f"({stats['near_dup_removed_pct']}%) clusters={stats['clusters']:,} "
          f"singletons={stats['singletons']:,}  ({time.time() - t0:.0f}s)", flush=True)

    # --- pass 2/2: write kept docs, PARALLEL over SHARDS ---
    # Each worker re-reads its shard, batch-tokenizes kept docs, routes by release bucket,
    # and writes its own <bucket>/<bucket>-s{shard_index:05d}.jsonl.gz (gzip level 4).
    # সাধু is already known from pass 1. Shard file boundaries differ from a single writer,
    # but the doc SET per bucket is identical.
    print(f"pass 2/2: writing kept docs + token counts over {len(spans):,} shards, "
          f"workers={args.workers} ...", flush=True)
    t_p3 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for b in ("releasable", "conditional", "train_only"):
        (outdir / b).mkdir(exist_ok=True)

    p3_jobs = []
    for (si, k, offset) in spans:
        s, _tr, path, nm = shards[si]
        p3_jobs.append((si, s, path, nm, str(outdir), k,
                        keep[offset:offset + k].copy(),
                        is_sadhu[offset:offset + k].copy()))
    del keep, is_sadhu

    if args.workers <= 1:
        _init_pass3(args.tokenizer)
        results = [_pass3_worker(job) for job in p3_jobs]
    else:
        import multiprocessing as _mp
        with _mp.Pool(args.workers, initializer=_init_pass3,
                      initargs=(args.tokenizer,)) as pool:
            results = list(pool.imap_unordered(_pass3_worker, p3_jobs))

    per_bucket: dict[str, dict] = {}
    total_tokens = 0
    for res in results:                          # aggregate per-shard tallies (order-free)
        for b, (dc, tk, sd) in res.items():
            pb = per_bucket.setdefault(b, {"docs": 0, "tokens": 0, "sadhu": 0})
            pb["docs"] += dc
            pb["tokens"] += tk
            pb["sadhu"] += sd
            total_tokens += tk
    print(f"  pass 2 done in {time.time() - t_p3:.1f}s", flush=True)

    report = {
        **stats,
        "per_bucket": per_bucket,
        "total_tokens": total_tokens,
        "total_tokens_B": round(total_tokens / 1e9, 4),
        "tokenizer": args.tokenizer,
        "tracks": args.tracks,
        "seconds": round(time.time() - t0, 1),
        "params": {"n_perm": n_perm, "bands": args.bands, "rows": args.rows,
                   "ngram": args.ngram, "seed": args.seed},
    }

    print("\n" + "=" * 72)
    print(f"docs_in={stats['docs_in']:,}  kept={stats['docs_kept']:,}  "
          f"near-dup removed={stats['near_dup_removed']:,} "
          f"({stats['near_dup_removed_pct']}%)")
    print(f"clusters={stats['clusters']:,}  singletons={stats['singletons']:,}  "
          f"total tokens={total_tokens:,} (~{report['total_tokens_B']}B)")
    print("per bucket (docs / ~Btok / sadhu):")
    for b in ("releasable", "conditional", "train_only"):
        pb = per_bucket.get(b, {"docs": 0, "tokens": 0, "sadhu": 0})
        print(f"  {b:12s} docs={pb['docs']:>11,} ~{pb['tokens'] / 1e9:>7.3f}B "
              f"sadhu={pb['sadhu']:>9,}")
    print(f"elapsed {report['seconds']}s")
    print("=" * 72)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
