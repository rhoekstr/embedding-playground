#!/usr/bin/env python3
"""
Fast analysis tool over the emitted slice (reads embeddings.json, no GloVe).
Used to hunt for clean PMC analogies and good Mode A "tricky neighbor" pairs.

  python3 build/explore.py
"""
import base64
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "embeddings.json").read_text())

words = [w["w"] for w in data["words"]]
cluster = {w["w"]: w["cluster"] for w in data["words"]}
pos = {w: i for i, w in enumerate(words)}
dim = data["meta"]["dim"]
blob = np.frombuffer(base64.b64decode(data["vectors_b64"]), dtype="<f4")
mat = blob.reshape(-1, dim).astype(np.float64)
norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True))


def nn(word, k=10):
    sims = norm @ norm[pos[word]]
    order = np.argsort(-sims)
    return [(words[j], float(sims[j])) for j in order if words[j] != word][:k]


def analogy(a, b, c, k=5):
    d = norm[pos[a]] - norm[pos[b]] + norm[pos[c]]
    d = d / (np.linalg.norm(d) or 1.0)
    sims = norm @ d
    order = np.argsort(-sims)
    excl = {a, b, c}
    out = []
    for j in order:
        if words[j] in excl:
            continue
        out.append((words[j], float(sims[j])))
        if len(out) == k:
            break
    return out


def show(a, b, c):
    top = analogy(a, b, c)
    print(f"  {a:12s}- {b:12s}+ {c:12s}-> {top[0][0]:14s} "
          f"[{', '.join(f'{w}({s:.2f})' for w, s in top)}]")


if __name__ == "__main__":
    print("=== PMC / DOL analogy candidates ===")
    pmc = [
        ("annual", "quarterly", "monthly"),
        ("measure", "quarterly", "annual"),
        ("milestone", "project", "program"),
        ("employer", "employee", "worker"),
        ("manager", "management", "employee"),
        ("management", "manager", "worker"),
        ("director", "division", "agency"),
        ("director", "department", "agency"),
        ("salary", "employee", "employer"),
        ("wage", "worker", "job"),
        ("funding", "grant", "budget"),
        ("unemployment", "unemployed", "employment"),
        ("training", "skills", "apprenticeship"),
        ("compliance", "regulation", "policy"),
        ("benchmark", "measure", "target"),
        ("metric", "measure", "indicator"),
        ("employee", "worker", "manager"),
        ("agency", "federal", "state"),
        ("federal", "agency", "state"),
        ("worker", "labor", "employment"),
        ("evaluation", "assessment", "review"),
        ("quarterly", "year", "annual"),
        ("monthly", "month", "year"),
        ("budget", "fiscal", "annual"),
        ("performance", "measure", "evaluation"),
        ("outcome", "output", "input"),
        ("employer", "hiring", "recruitment"),
        ("retirement", "pension", "salary"),
    ]
    for a, b, c in pmc:
        if all(x in pos for x in (a, b, c)):
            show(a, b, c)
        else:
            miss = [x for x in (a, b, c) if x not in pos]
            print(f"  (skip {a}-{b}+{c}: missing {miss})")

    print("\n=== Mode A 'tricky neighbors': high cosine, different meaning ===")
    # candidate pairs the PRD suggests + nearby
    pairs = [
        ("monitor", "evaluate"), ("table", "chart"), ("measure", "metric"),
        ("table", "scale"), ("plant", "plant"), ("bank", "fair"),
        ("light", "fair"), ("scale", "measure"), ("rate", "rank"),
        ("current", "trend"), ("plan", "program"), ("policy", "procedure"),
        ("seal", "stamp"), ("bat", "ball"), ("spring", "season"),
    ]
    for a, b in pairs:
        if a in pos and b in pos:
            sims = float(norm[pos[a]] @ norm[pos[b]])
            print(f"  {a:10s} <-> {b:10s} cos={sims:.2f}  "
                  f"(clusters: {cluster[a]} / {cluster[b]})")

    print("\n=== Mode C neighborhoods (within-slice) ===")
    for w in ["measure", "bank", "average", "performance", "compliance"]:
        print(f"  {w}: {', '.join(x for x, _ in nn(w, 8))}")

    print("\n=== Polysemy: within-slice + true (full-space) neighbors ===")
    tn = data["true_neighbors"]
    poly = [w["w"] for w in data["words"] if w["cluster"] == "c6_polysemy"]
    for w in poly:
        within = ", ".join(x for x, _ in nn(w, 6))
        true = ", ".join(tn[w][:8])
        print(f"  {w:9s} within: {within}")
        print(f"  {'':9s} true:   {true}")
