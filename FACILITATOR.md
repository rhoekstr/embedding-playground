# Facilitator notes — Embedding Playground (PMC AI Coffee, Session 2)

One page. The tool teaches three claims (PRD §3); your job is to listen for the room
re-stating them unprompted. If people are *playing* instead of saying "oh, that's why
my prompts about 'metrics' and 'measures' get different answers," steer back to the
pre-baked scenarios.

## Getting the tool to people

- **Primary: it's hosted.** Point everyone at **https://embeddings.awrylabs.com**
  (GitHub Pages, public repo `rhoekstr/embedding-playground`). Nothing to deploy at
  DOL — participants just open the URL in Chrome or Edge.
- **Offline fallback:** `dist/embedding-playground.html` (~1.4 MB, in the repo too).
  One self-contained file that opens by double-click with no network. Keep a copy on
  a USB stick or shared drive in case the DOL proxy blocks the site on the day.
- Either way it never phones home. All math runs in the browser; refresh resets
  everything; nothing is saved.
- To publish changes: edit, rebuild if the lexicon changed, `git push` — Pages
  redeploys in about a minute.

## Run of show (~10 minutes, after the 5-min embeddings lesson)

| min | do | say / listen for |
|-----|----|------------------|
| 0–1 | Everyone opens the URL. Point at the four tabs. | "Four questions, four tabs. Start on A." |
| 1–3 | Mode A: have pairs run **Colors & relatives** (guaranteed win), then **Synonyms for "measure."** | Listen for claim 1: *close in the geometry = behaves alike.* The "measure/metric/indicator" density IS why phrasing matters. |
| 3–4 | Mode A: **PM vs DOL vocabulary.** | The PMC payoff: domain boundaries are encoded. Generic LLMs don't naturally speak PMC's language. |
| 4–6 | Mode B: **king − man + woman**. Let them watch the animation. | The audible-reaction moment. Then **paris − france + japan** to show it's mechanism, not magic. |
| 6–7 | Mode B: one PMC analogy (**worker − labor + employer**), then the **failure button**. | "The failures are the lesson too — this is a tool with edges, not an oracle." |
| 7–8 | Mode B: **doctor − man + woman** (the designed dead heat), then **the gender axis**. | Use the one-liner below, verbatim if needed. Don't dwell; two data points then move. |
| 8–9 | Mode D: **"measure"** — the legislative cluster hugs the center, the KPI cluster floats away (68/32). Then **"court"**, then type **wedding** into "Add context" and watch the dating cluster rush in (15% → 33%, takes the lead). **bank + river** flips outright (38% → 65%). | "Your daily KPI word is mostly a *legislative* word to the model — say 'metric' when you mean the KPI sense. And watch: adding ONE context word swings which meaning wins. Doing this by hand is exactly what next week's contextual models do automatically." |
| 9–10 | Mode C: **"average"**, then free exploration (or Mode D "bat"/"organ" for fun). | Thesis callback: "this neighborhood is the generic center — next week is about how we condition the model *out* of it." |

**Mode D in one breath:** every word gets ONE vector no matter how many meanings it
has. The word sits at the center; each sense's words sit at their true cosine
distance — the cluster hugging the center is the sense the training text fed most.
"measure" = 68% legislative; "bat" = 64% baseball (the animal loses!); "court" has
four spokes — legal at 47% with dating way out at 15%; "organ" instrument/body is a
dead heat. A sense parked far away (court-dating, bill-as-beak, date-the-fruit at
9%) barely exists to the model: the corpus doesn't just rank senses, it decides
which senses exist. Cleanest bridge to next week: contextual embeddings exist
precisely because one-vector-per-word collapses meanings.

## The bias one-liner (have it ready, say it calmly)

> "This isn't the model 'thinking' anything; it's the geometry reflecting how these
> words appeared together in the text it was trained on. That's the whole insight —
> and it's why output-quality verification is your job, not the model's."

Framing rule: descriptive, not normative. "The model learned this from human writing,"
never "the model is sexist." The tool's own captions already follow this.

**The doctor button is a designed dead heat — lean into it.** The lexicon now includes
"physician" on purpose, so *doctor − man + woman* shows **physician 0.61** and
**nurse 0.61** with two dotted hops — a statistical tie between the gender-neutral
synonym and the stereotyped role. The discussion move: "the math doesn't pick nurse —
it lands exactly between the synonym and the stereotype. The stereotype being *that*
close is the finding." If the room wants the clean single-answer version, the
**ceo − man + woman → executive** button is decisive (0.62, next candidate 0.37),
and the gender-axis view shows the systematic skew.

## If something flops live

- **An analogy resolves to a weird word:** that's PRD §12's teachable moment. Say so:
  "this is what a 500-word slice plus 2010-era embeddings does — the math is real,
  the tool is a simplification." The footer says the same thing.
- **Someone's browser won't open it:** pair them up. The activity is designed for pairs anyway.
- **The room is too quiet:** drop to Mode A scenario 5 (Colors) — it's the confidence builder.
- **A skeptic replicates an analogy online and gets a different answer.** One known case
  (all eight headline buttons were verified against the full 400K-word GloVe space):
  - *bigger − big + small*: full space says "larger" (0.78) over "smaller" (0.77) — "larger"
    is deliberately not in our lexicon so the grammar demo stays clean. If raised: the
    geometry encodes "comparative-ness" more strongly than the big↔small polarity.
  - The doctor/physician dead heat is no longer a surprise — it's in the tool by design
    (see the bias section above).
  - Everything else (king/queen, tokyo, employee, monitoring, executive, running) is the
    exact full-space winner.

## Day-before checklist

- [ ] Open **https://embeddings.awrylabs.com** from an actual DOL machine on the DOL network (Chrome **and** Edge; Firefox too if it's on the image). This is the critical check — if the proxy blocks it, fall back to the offline file.
- [ ] Click through all 7 Mode A scenarios, all 8 Mode B buttons, the failure button, the gender axis, all 5 Mode C words, and at least measure/court/organ in Mode D's 15.
- [ ] Preview the bias demo with one trusted colleague (PRD §12) — check the framing lands.
- [ ] Have `dist/embedding-playground.html` on a USB stick or shared drive as the offline fallback, and confirm it opens by double-click on a DOL machine.
- [ ] If you edit the lexicon before then: `python3 build/preprocess.py && python3 build/make_dist.py`, re-check the Mode B buttons still resolve, then `git push` to redeploy.

## The three claims you're listening for (success = the room says these unprompted)

1. Words have positions, and "close" = "behaves alike in the model."
2. Different phrasings sit in different positions — that's why paraphrasing changes the answer.
3. The geometry was learned from human text, including human prejudice — it's baked in.
