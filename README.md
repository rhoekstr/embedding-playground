# Embedding Playground

An interactive demonstration that **meaning is geometry** — and that "close in
that geometry" is why paraphrasing a prompt changes a language model's answer.

**Live:** https://embeddings.awrylabs.com

Four modes, one question each:

- **A — How close are these?** An N×N grid of cosine similarities between words
  you pick. Jargon clusters densely; that density is why phrasing matters.
- **B — Word math.** Analogy arithmetic (`king − man + woman ≈ queen`) animated
  as actual operations: subtraction draws the difference arrow, addition
  visibly carries it to the third word, and a dotted hop lands on the nearest
  real word — with statistical ties drawn as *two* hops rather than a fake
  single winner (`doctor − man + woman` is a designed dead heat between
  *physician* and *nurse*; the closeness is the lesson). Includes analogies
  that *fail*, on purpose — and a look at how occupations project onto the
  he–she axis, a learned-bias pattern the model absorbed from human writing.
- **C — What's near this word?** A radial map of a word's neighborhood. Click
  any boxed neighbor to walk through the space; each word's *true* top
  neighbors from the full 400K-word vocabulary are shown alongside for
  honesty.
- **D — Which meaning wins?** A many-sense word sits at the center; each sense
  (up to four — *court* has legal, royal, sports, and dating) gets a spoke,
  with its anchor words at their true cosine distance. The closest cluster is
  the sense the training text fed most; a sense parked at the rim barely
  exists to the model. Then **add context words** and watch the distribution
  shift live: `bank + river` flips the dominance to the riverside;
  `court + wedding` pulls the dating sense from last place to first. Composing
  context by hand is a primitive version of what contextual embeddings do
  automatically.

Everything runs client-side from a curated 755-word slice of GloVe (six
thematic clusters, 35 pre-baked teaching scenarios). No backend, no logging,
no network calls after load. The single offline file in [`dist/`](dist/) is
the same tool and opens directly from disk.

## Honest limitations (shown in the tool, repeated here)

The 2D pictures are flattenings of 300-dimensional vectors — the math is exact,
the pictures are approximate. "Nearest neighbor within the slice" is not nearest
in the full 400K-word space, so each word's true top neighbors from the full
space are shown as context. The bias demonstrations are descriptive: the
geometry reflects how words co-occur in human writing.

## Teaching with it

The tool was built for a ~10-minute hands-on segment in a workplace
AI-literacy series. [FACILITATOR.md](FACILITATOR.md) is the one-page run of
show: which scenarios to run in what order, how to frame the bias
demonstration (descriptive, not normative), what to say when an analogy
flops, and a day-before checklist.

## Rebuilding the data

```
# 1. get GloVe 6B (Stanford NLP) and unzip glove.6B.300d.txt into data/
# 2. regenerate the word slice, neighbors, and verified analogies:
python3 build/preprocess.py
# 3. repackage the single-file offline version:
python3 build/make_dist.py
```

The lexicon (six clusters: universal anchors, kinship, geography, performance
management, bias set, polysemy) lives in `build/lexicon.py`, along with the
sense-anchor sets that drive Mode D. The build report
(`build/build_report.txt`) verifies every curated analogy resolves and every
sense anchor exists within the slice before it ships, and records each
multi-sense word's dominance split.

## Credits

Word vectors: [GloVe](https://nlp.stanford.edu/projects/glove/) — Jeffrey
Pennington, Richard Socher, Christopher D. Manning (2014), *GloVe: Global
Vectors for Word Representation*, EMNLP. Vectors released under the
[PDDL](https://opendatacommons.org/licenses/pddl/).

Plain HTML/CSS/JS. No frameworks, no build step, no math libraries — cosine
similarity is fifteen lines of JavaScript.
