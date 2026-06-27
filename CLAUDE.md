# CLAUDE.md

Canonical, version-controlled guide for working in this repo. Keep it in sync with the code,
[`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) (specs + website copy), and
[`FACILITATOR.md`](./FACILITATOR.md) (teaching run-of-show).

## What this is
**Embedding Playground** — an interactive, client-side demo that *meaning is geometry* (live at
**embeddings.awrylabs.com**). Four modes (similarity grid · word math · neighborhood walk · sense
competition) over a curated 755-word GloVe slice. Built for a ~10-minute AI-literacy teaching segment.

## The rules that can't bend (READ FIRST)
1. **No frameworks, no build step, no runtime dependencies.** Plain HTML/CSS/JS — cosine similarity is
   ~15 lines. The app (`index.html`, `app.js`, `embeddings.js`, `embeddings.json`) runs by opening the
   file. Don't introduce a framework, a bundler, or a math library.
2. **All computation is client-side, and it stays private.** No backend, no analytics, no cookies, no
   network calls after page load — refresh resets everything. Keep it that way (Awry brand + the demo's
   own stated promise).
3. **The offline single-file build must equal the live tool.** `dist/embedding-playground.html` is
   byte-for-byte the same tool, openable from disk. Regenerate it via the build step after any change —
   **never hand-edit `dist/`.**

## Data pipeline (Python, build-time only — the app itself has no build)
```bash
# 1. drop GloVe 6B (Stanford NLP) glove.6B.300d.txt into data/
python3 build/preprocess.py    # regenerate the word slice, true neighbors, verified analogies
python3 build/make_dist.py      # repackage the single-file offline version
```
The lexicon + sense-anchor sets live in `build/lexicon.py`; `build/build_report.txt` verifies every
curated analogy resolves and every sense anchor exists in the slice **before it ships** — don't ship a
slice that fails the report.

## Conventions / invariants
- **Honesty by design:** the UI labels its own simplifications (2D is a flattening of 300-D; within-slice
  neighbors are shown against true full-vocab neighbors; near-ties draw two hops, never a fake winner).
  Preserve this when editing any mode.
- **Accessibility:** similarity is always a number (never color alone); every drag has a click/keyboard
  equivalent; clickable words are boxed consistently in every mode.
- **Bias demos are descriptive, not normative** ("learned from how words co-occur in human writing").
- **Visual identity:** cream ground (not white); cosine ramp cream → blue-black (deliberately not
  viridis/plasma); serif for the words, geometric sans for chrome, tabular figures; no icons.

## Docs
- **CLAUDE.md** (this) — invariants, data pipeline, conventions.
- **README.md** — the front door (modes, limitations, rebuild).
- **PROJECT_SUMMARY.md** — specs + lift-ready website copy.
- **FACILITATOR.md** — teaching run-of-show.
