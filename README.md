# Embedding Playground

An interactive demonstration that **meaning is geometry** — and that "close in
that geometry" is why paraphrasing a prompt changes a language model's answer.

**Live:** https://embeddings.awrylabs.com

Three modes, one question each:

- **A — How close are these?** An N×N grid of cosine similarities between words
  you pick. Jargon clusters densely; that density is why phrasing matters.
- **B — Word math.** Analogy arithmetic (`king − man + woman ≈ queen`) with the
  vector arithmetic animated in a 2D projection. Includes analogies that *fail*,
  on purpose — and a look at how occupations project onto the he–she axis,
  a learned-bias pattern the model absorbed from human writing.
- **C — What's near this word?** A radial map of a word's neighborhood. Click a
  neighbor to walk through the space. Polysemous words ("plant") show their
  senses collapsed into one vector — the limitation that motivated contextual
  embeddings.

Everything runs client-side from a curated 492-word slice of GloVe. No backend,
no logging, no network calls after load. The single offline file in
[`dist/`](dist/) is the same tool and opens directly from disk.

## Honest limitations (shown in the tool, repeated here)

The 2D pictures are flattenings of 300-dimensional vectors — the math is exact,
the pictures are approximate. "Nearest neighbor within 492 words" is not nearest
in the full 400K-word space, so each word's true top neighbors from the full
space are shown as context. The bias demonstrations are descriptive: the
geometry reflects how words co-occur in human writing.

## Rebuilding the data

```
# 1. get GloVe 6B (Stanford NLP) and unzip glove.6B.300d.txt into data/
# 2. regenerate the word slice, neighbors, and verified analogies:
python3 build/preprocess.py
# 3. repackage the single-file offline version:
python3 build/make_dist.py
```

The lexicon (six clusters: universal anchors, kinship, geography, performance
management, bias set, polysemy) lives in `build/lexicon.py`. The build report
(`build/build_report.txt`) verifies every curated analogy resolves within the
slice before it ships.

## Credits

Word vectors: [GloVe](https://nlp.stanford.edu/projects/glove/) — Jeffrey
Pennington, Richard Socher, Christopher D. Manning (2014), *GloVe: Global
Vectors for Word Representation*, EMNLP. Vectors released under the
[PDDL](https://opendatacommons.org/licenses/pddl/).

Plain HTML/CSS/JS. No frameworks, no build step, no math libraries — cosine
similarity is fifteen lines of JavaScript.
