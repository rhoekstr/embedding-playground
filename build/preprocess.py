#!/usr/bin/env python3
"""
Preprocessing pipeline for the Embedding Playground (PRD section 6).

  1. Load GloVe 6B 300d (full 400K vocab).
  2. Filter to the curated lexicon; report any missing words.
  3. Compute each lexicon word's true top-20 nearest neighbors from the FULL
     space (store neighbor strings only).
  4. Verify the candidate analogies resolve within the 500-word slice.
  5. Tag each word with its cluster.
  6. Emit embeddings.json (vectors as a base64 Float32Array blob).

Run:  python3 build/preprocess.py
"""

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lexicon import (  # noqa: E402
    ANALOGY_CANDIDATES,
    CLUSTERS,
    FAILURE_CANDIDATES,
)

ROOT = Path(__file__).resolve().parent.parent
GLOVE_PATH = ROOT / "data" / "glove.6B.300d.txt"
OUT_PATH = ROOT / "embeddings.json"
REPORT_PATH = ROOT / "build" / "build_report.txt"

DIM = 300
NEIGHBOR_K = 20          # true neighbors from full space
ANALOGY_TOPK = 5         # nearest matches within slice
MIN_VERIFIED = 15        # PRD: ~15 analogies that resolve within the slice


def log(msg, lines):
    print(msg)
    lines.append(msg)


def build_word_to_cluster():
    """Flatten CLUSTERS, asserting each word appears in exactly one cluster."""
    w2c = {}
    dupes = []
    for cluster, words in CLUSTERS.items():
        for w in words:
            if w in w2c:
                dupes.append((w, w2c[w], cluster))
            else:
                w2c[w] = cluster
    return w2c, dupes


def load_glove(path):
    """Load full GloVe into (words list, float32 matrix [N, DIM])."""
    print(f"Loading GloVe from {path} ...")
    words = []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != DIM + 1:
                continue
            words.append(parts[0])
            rows.append(parts[1:])
    mat = np.array(rows, dtype=np.float32)
    print(f"  loaded {mat.shape[0]:,} words x {mat.shape[1]} dims")
    return words, mat


def normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main():
    report = []
    t0 = time.time()

    if not GLOVE_PATH.exists():
        sys.exit(f"GloVe file not found at {GLOVE_PATH}")

    w2c, dupes = build_word_to_cluster()
    if dupes:
        log("ERROR: words assigned to multiple clusters:", report)
        for w, c1, c2 in dupes:
            log(f"  '{w}' in both {c1} and {c2}", report)
        sys.exit("Fix duplicate cluster assignments in lexicon.py and re-run.")

    lexicon_words = list(w2c.keys())
    log(f"Lexicon size (pre-filter): {len(lexicon_words)} words", report)

    full_words, full_mat = load_glove(GLOVE_PATH)
    full_index = {w: i for i, w in enumerate(full_words)}
    full_norm = normalize(full_mat)

    # --- 2. filter to lexicon, report misses ---
    present = [w for w in lexicon_words if w in full_index]
    missing = [w for w in lexicon_words if w not in full_index]
    log("", report)
    log(f"Lexicon words present in GloVe: {len(present)}", report)
    if missing:
        log(f"MISSING from GloVe ({len(missing)}) - dropped, consider substitutes:", report)
        # group missing by cluster
        by_cluster = {}
        for w in missing:
            by_cluster.setdefault(w2c[w], []).append(w)
        for c, ws in by_cluster.items():
            log(f"  {c}: {', '.join(ws)}", report)
    else:
        log("All lexicon words present in GloVe.", report)

    # cluster size summary (post-filter)
    log("", report)
    log("Cluster sizes (post-filter):", report)
    csizes = {}
    for w in present:
        csizes[w2c[w]] = csizes.get(w2c[w], 0) + 1
    for c in CLUSTERS:
        log(f"  {c}: {csizes.get(c, 0)}", report)

    # --- slice matrices ---
    slice_words = present
    slice_idx = np.array([full_index[w] for w in slice_words])
    slice_mat = full_mat[slice_idx]
    slice_norm = full_norm[slice_idx]
    slice_pos = {w: i for i, w in enumerate(slice_words)}

    # --- 3. true top-20 neighbors from full space (batched) ---
    print("\nComputing true neighbors from full space ...")
    neighbors = {}
    BATCH = 64
    for start in range(0, len(slice_words), BATCH):
        qwords = slice_words[start:start + BATCH]
        q = full_norm[[full_index[w] for w in qwords]]      # (b, DIM)
        sims = q @ full_norm.T                               # (b, N)
        for r, w in enumerate(qwords):
            order = np.argpartition(-sims[r], NEIGHBOR_K + 1)[:NEIGHBOR_K + 1]
            order = order[np.argsort(-sims[r][order])]
            neigh = [full_words[j] for j in order if full_words[j] != w][:NEIGHBOR_K]
            neighbors[w] = neigh
    print(f"  done ({len(neighbors)} words)")

    # --- 4. analogy verification within slice ---
    def analogy_topk(a, b, c, k=ANALOGY_TOPK):
        """a - b + c, nearest within slice (normalized vectors), excl inputs."""
        va, vb, vc = (slice_norm[slice_pos[x]] for x in (a, b, c))
        d = va - vb + vc
        d = d / (np.linalg.norm(d) or 1.0)
        sims = slice_norm @ d
        order = np.argsort(-sims)
        out = []
        excl = {a, b, c}
        for j in order:
            w = slice_words[j]
            if w in excl:
                continue
            out.append((w, float(sims[j])))
            if len(out) == k:
                break
        return out

    log("", report)
    log("=" * 60, report)
    log("ANALOGY VERIFICATION (nearest within the 500-word slice)", report)
    log("=" * 60, report)

    verified = []
    for cand in ANALOGY_CANDIDATES:
        a, b, c = cand["a"], cand["b"], cand["c"]
        inputs_ok = all(x in slice_pos for x in (a, b, c))
        if not inputs_ok:
            missing_in = [x for x in (a, b, c) if x not in slice_pos]
            log(f"[SKIP] {cand['id']:12s} {a}-{b}+{c}: inputs not in slice: {missing_in}", report)
            continue
        top = analogy_topk(a, b, c)
        top_str = ", ".join(f"{w}({s:.2f})" for w, s in top)
        exp = cand.get("expected")
        result_word = top[0][0]
        if exp is None:
            status = "OPEN"  # PMC analogies with no fixed expected answer
            ok = True
        else:
            rank = next((i for i, (w, _) in enumerate(top) if w == exp), None)
            if rank == 0:
                status, ok = "PASS", True
            elif rank is not None:
                status, ok = f"PARTIAL(rank {rank+1})", True
            else:
                status, ok = "FAIL", False
        log(f"[{status:14s}] {cand['id']:12s} {a}-{b}+{c} -> {result_word}", report)
        log(f"                 top5: {top_str}", report)
        if exp is not None:
            log(f"                 expected: {exp}", report)
        cand_out = {
            "id": cand["id"], "a": a, "b": b, "c": c,
            "expected": exp, "teaches": cand["teaches"],
            "headline": cand.get("headline", False),
            "kind": cand.get("kind", "headline"),
            "label": cand.get("label", f"{a} - {b} + {c}"),
            "result": result_word,
            "top5": [{"w": w, "sim": round(s, 4)} for w, s in top],
            "status": status,
        }
        if ok:
            verified.append(cand_out)

    log("", report)
    log(f"Verified/usable analogies: {len(verified)} (target >= {MIN_VERIFIED})", report)
    if len(verified) < MIN_VERIFIED:
        log("WARNING: fewer than target verified analogies; add more candidates.", report)

    # --- failure analogies (evaluate, don't filter) ---
    log("", report)
    log("FAILURE-EXAMPLE candidates (for the Mode B 'failure' button):", report)
    failure_eval = []
    for cand in FAILURE_CANDIDATES:
        a, b, c = cand["a"], cand["b"], cand["c"]
        if not all(x in slice_pos for x in (a, b, c)):
            log(f"  [SKIP] {cand['id']}: inputs not all in slice", report)
            continue
        top = analogy_topk(a, b, c)
        top_str = ", ".join(f"{w}({s:.2f})" for w, s in top)
        log(f"  {cand['id']:14s} {a}-{b}+{c} -> {top[0][0]}   [{top_str}]", report)
        failure_eval.append({
            "id": cand["id"], "a": a, "b": b, "c": c,
            "teaches": cand["teaches"], "result": top[0][0],
            "top5": [{"w": w, "sim": round(s, 4)} for w, s in top],
        })

    # --- 6. emit embeddings.json ---
    # Vectors as one base64 little-endian Float32Array (row-major, one row per
    # word in 'words' order). Full GloVe precision: the slice is only ~500 words,
    # so file size isn't a constraint and we ship exact float32 rather than
    # quantizing. Cosine/analogy math normalizes per-vector in the browser.
    vec_blob = slice_mat.astype("<f4").tobytes()
    vec_b64 = base64.b64encode(vec_blob).decode("ascii")

    words_out = [{"w": w, "cluster": w2c[w]} for w in slice_words]
    neighbors_out = {w: neighbors[w] for w in slice_words}

    payload = {
        "meta": {
            "source": "GloVe 6B 300d (Stanford NLP)",
            "dim": DIM,
            "count": len(slice_words),
            "note": "Vectors are raw GloVe rows packed as a little-endian "
                    "Float32Array (row-major, one row per word in 'words' "
                    "order), base64-encoded. Cosine and analogy math normalize "
                    "per-vector in the browser. true_neighbors are from the full "
                    "400K-word space; only this 500-word slice ships.",
        },
        "clusters": list(CLUSTERS.keys()),
        "words": words_out,
        "vectors_b64": vec_b64,
        "true_neighbors": neighbors_out,
        "curated_analogies": verified,
        "failure_candidates": failure_eval,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    # Also emit embeddings.js (a script-tag global) so the tool works from
    # file:// where Chrome blocks fetch() of local JSON.
    js_path = ROOT / "embeddings.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.EMBEDDINGS=")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";")

    size_kb = OUT_PATH.stat().st_size / 1024
    log("", report)
    log(f"Wrote {OUT_PATH} ({size_kb:.0f} KB raw; ~{size_kb*0.35:.0f} KB gzipped est.)", report)
    log(f"  words: {len(slice_words)}  vectors: {len(slice_words)}x{DIM} float32", report)
    log(f"Total time: {time.time()-t0:.1f}s", report)

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
