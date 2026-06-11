"""
Lexicon for the Embedding Playground (PMC AI Coffee, Session 2).

Source of truth for the curated ~500-word GloVe slice. Every word must be a
single lowercase token (GloVe 6B is tokenized + lowercased). preprocess.py
checks each word against GloVe and reports any that are missing so they can be
fixed or dropped.

Clusters (PRD section 5):
  c1_universal  - universal anchors (colors, animals, foods, weather, emotions,
                  body, household, activities, common adjectives)
  c2_kinship    - kinship, people, pronouns, age/relational terms
  c3_geography  - countries, capitals, regions
  c4_pm_dol     - performance-management + DOL vocabulary   [REVIEW: Robert]
  c5_bias       - bias-demonstration occupations + descriptors [REVIEW: Robert]
  c6_polysemy   - polysemy traps (one word, two senses)

NOTE: clusters c4_pm_dol and c5_bias are first-draft and flagged for Robert's
review per PRD sections 5 and 12. Edit the lists below and re-run preprocess.py.
"""

CLUSTERS = {
    # ------------------------------------------------------------------ c1
    "c1_universal": [
        # colors
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "navy", "crimson", "scarlet", "teal", "violet",
        # animals
        "dog", "cat", "horse", "cow", "pig", "sheep", "lion", "tiger", "bear",
        "wolf", "fox", "rabbit", "mouse", "bird", "fish", "snake", "frog",
        "deer", "elephant", "monkey",
        # foods
        "bread", "cheese", "rice", "apple", "banana", "egg", "meat", "milk",
        "coffee", "tea", "sugar", "salt", "cake", "fruit",
        # weather
        "rain", "snow", "wind", "storm", "cloud", "sun", "fog", "heat",
        "cold", "ice",
        # emotions
        "happy", "sad", "angry", "afraid", "joy", "fear", "love", "hate",
        "calm", "excited",
        # body
        "head", "hand", "foot", "eye", "arm", "leg", "heart", "face",
        "finger", "ear",
        # household
        "table", "chair", "door", "window", "bed", "lamp", "cup", "plate",
        "knife", "spoon", "clock", "book",
        # activities (incl. base+ing for the morphology analogy)
        "running", "walking", "hiking", "swimming", "jogging", "climbing",
        "reading", "writing", "cooking", "dancing", "walk", "run",
        # common adjectives (for grammar/morphology analogies)
        "big", "small", "bigger", "smaller", "good", "bad", "fast", "slow",
        "old", "young", "new", "large",
        # 2026-06-11 relationship-mining additions (true-neighbor stitch words)
        "biking", "bright", "dark", "flour", "butter", "huge", "pace", "quiet",
        "shoulder",
        # 2026-06-11 deep-mining round 2 (target ~800 lexicon)
        "chocolate", "cream", "vegetables", "beef", "peanut", "pepper", "salmon",
        "wheat", "corn", "pork", "juice", "chicken", "mango", "coconut", "dairy",
        "goat", "rat", "squirrel", "leopard", "cattle", "coyote", "elk", "pet",
        "animal", "wrist", "elbow", "ankle", "knee", "thigh", "brain", "nose",
        "warm", "mist", "temperature", "hurricane", "snowfall", "cool",
        "winter", "summer", "autumn", "scared", "worry", "delight", "glad",
        "lavender", "massive", "enormous", "kitchen",
        "football", "soccer", "ball", "game", "team", "beach",
    ],
    # ------------------------------------------------------------------ c2
    "c2_kinship": [
        "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
        "father", "mother", "son", "daughter", "brother", "sister", "uncle",
        "aunt", "nephew", "niece", "grandfather", "grandmother", "husband",
        "wife", "child", "adult", "elder", "baby", "parent", "sibling",
        "cousin", "family", "men", "women", "he", "she", "him", "her", "his",
        "boys", "girls", "gentleman", "lady", "male", "female", "person",
        "people", "friend", "neighbor", "stranger", "teenager", "infant",
        "toddler", "youth", "senior", "widow", "bride", "groom", "twin",
        "heir", "lord", "ruler", "citizen", "kid", "grandson", "granddaughter",
        "spouse", "ancestor", "generation",
        # 2026-06-11 relationship-mining additions
        "newborn",
        # 2026-06-11 deep-mining round 2
        "girlfriend", "boyfriend", "mom", "dad", "married", "wedding",
        "pregnant", "teen", "junior", "youngster", "offspring", "descendant",
        "couple", "divorced", "monarch", "throne", "crown", "royal", "duke",
        "duchess", "empress", "paternal",
    ],
    # ------------------------------------------------------------------ c3
    "c3_geography": [
        # countries
        "france", "japan", "germany", "italy", "spain", "china", "russia",
        "india", "england", "canada", "mexico", "brazil", "egypt", "greece",
        "portugal", "poland", "norway", "sweden", "ireland", "austria",
        "finland", "denmark", "netherlands", "turkey",
        # capitals (matched to countries above where possible)
        "paris", "tokyo", "berlin", "rome", "madrid", "beijing", "moscow",
        "london", "ottawa", "athens", "lisbon", "warsaw", "oslo", "stockholm",
        "dublin", "vienna", "cairo", "delhi", "helsinki", "copenhagen",
        "amsterdam", "ankara",
        # regions / geography-generic
        "europe", "asia", "africa", "america", "continent", "nation",
        "country", "city", "capital", "region", "border",
        # 2026-06-11 relationship-mining additions (capital/country pairs kept whole)
        "prague", "belgium", "brussels", "hungary", "budapest",
        # 2026-06-11 deep-mining round 2
        "romania", "bulgaria", "istanbul", "switzerland", "britain", "barcelona",
        "slovakia", "wales", "montreal", "argentina", "munich", "estonia",
        "lithuania", "scotland", "ukraine", "kiev", "frankfurt", "cyprus",
        "seoul", "korea", "shanghai", "luxembourg", "iceland", "australia",
        "venezuela", "taiwan", "toronto", "latvia", "colombia", "mumbai",
        "osaka", "taipei", "zurich", "croatia", "milan", "peru",
    ],
    # ------------------------------------------------------------------ c4
    # [REVIEW: Robert] Performance-management + DOL vocabulary.
    "c4_pm_dol": [
        # generic PM vocabulary
        "measure", "measurement", "target", "milestone", "baseline", "outcome",
        "output", "indicator", "metric", "benchmark", "quarterly", "fiscal",
        "performance", "evaluation", "monitoring", "monitor", "evaluate",
        "accountability", "compliance", "stakeholder", "deliverable", "scope",
        "goal", "objective", "result", "impact", "efficiency", "effectiveness",
        "productivity", "quality", "standard", "criteria", "assessment",
        "analysis", "analyst", "data", "report", "review", "audit", "oversight",
        "governance", "strategy", "planning", "budget", "cost", "resource",
        "timeline", "schedule", "progress", "status", "risk", "priority",
        "threshold", "variance", "trend", "forecast", "projection", "estimate",
        "average", "annual", "monthly", "weekly", "deadline", "plan", "project",
        "process", "procedure", "policy", "framework", "system", "value",
        "ratio", "percentage", "rate", "score", "gauge", "tracking",
        "alignment", "capacity", "workload", "completion", "achievement",
        "chart",
        # DOL-flavored vocabulary
        "agency", "program", "initiative", "mission", "strategic", "management",
        "workforce", "employment", "labor", "occupational", "training",
        "apprenticeship", "wage", "unemployment", "employer", "employee", "job",
        "worker", "hiring", "recruitment", "retention", "skills",
        "certification", "credential", "qualification", "vacancy", "position",
        "salary", "benefits", "pension", "retirement", "safety", "regulation",
        "enforcement", "inspection", "violation", "penalty", "grant", "funding",
        "appropriation", "federal", "state", "local", "jurisdiction", "statute",
        "mandate", "authority", "department", "division", "bureau", "office",
        "administration", "director", "official", "contractor", "vendor",
        "procurement", "eligibility", "applicant", "beneficiary", "claimant",
        "disability", "veteran", "minimum", "overtime", "union", "bargaining",
        "sector", "industry", "economy",
        # 2026-06-11 relationship-mining additions
        "government", "transparency", "implementation", "yearly", "assess",
        "index", "percent", "pay", "spending", "investment", "deputy",
        "legislation", "provision", "factory", "production",
        # 2026-06-11 deep-mining round 2
        "development", "growth", "inflation", "deficit", "reform", "law",
        "amendment", "regulatory", "payment", "earnings", "expenditure",
        "financing", "logistics", "internship", "negotiation", "supervision",
        "protection", "ministry", "maximum", "eligible", "comply", "implement",
        "examine", "improve", "achieve", "proposal", "requirement",
        "information", "education", "business", "construction", "manufacturing",
        "exports", "competitiveness", "accreditation", "timetable", "jobless",
        "efficacy", "accomplishment", "economic",
    ],
    # ------------------------------------------------------------------ c5
    # [REVIEW: Robert] Bias-demonstration set. No slurs, no named ethnicities or
    # religions (PRD section 7). Neutral descriptors that nonetheless carry
    # learned bias.
    "c5_bias": [
        # occupations across the gender/role spectrum
        "doctor", "nurse", "engineer", "teacher", "secretary", "ceo", "janitor",
        "programmer", "scientist", "homemaker", "firefighter", "librarian",
        "lawyer", "judge", "professor", "pilot", "soldier", "mechanic",
        "plumber", "carpenter", "electrician", "architect", "accountant",
        "banker", "manager", "executive", "clerk", "receptionist", "cashier",
        "chef", "maid", "nanny", "hairdresser", "designer", "artist",
        "musician", "athlete", "surgeon", "dentist", "therapist", "counselor",
        "midwife", "captain", "developer", "researcher", "technician",
        # gendered / loaded descriptors
        "masculine", "feminine", "manly", "strong", "weak", "emotional",
        "rational", "ambitious", "nurturing", "aggressive", "gentle",
        "assertive", "bossy", "sensitive", "logical", "caring", "dominant",
        "submissive", "independent", "beautiful", "handsome", "pretty",
        "tough",
        # 2026-06-11 relationship-mining additions — REVIEW: bias cluster is hand-curated
        "assistant", "bookkeeper", "housekeeper", "waitress", "pharmacist",
        "bartender", "welder", "chief", "charming", "businessman", "businesswoman",
        # 2026-06-11 deep-mining round 2 — REVIEW (bias cluster).
        # "physician" is a DESIGNED demo-break: doctor-man+woman now lands in a
        # dead heat between physician and nurse — discussion fodder, on purpose.
        "physician", "waiter", "entrepreneur", "schoolteacher", "chairman",
        "chairwoman", "attorney", "housewife", "physicist", "psychologist",
        "psychiatrist", "pediatrician", "stylist", "sculptor", "singer",
        "songwriter", "builder", "instructor", "blacksmith", "painter",
        "financier", "consultant", "auditor", "policeman", "policewoman",
        "apprentice", "advisor", "practitioner", "composer", "coach",
        "biologist", "fireman", "coordinator", "politician", "lecturer",
        "elegant", "polite",
    ],
    # ------------------------------------------------------------------ c6
    "c6_polysemy": [
        "bank", "spring", "bat", "rock", "plant", "light", "current", "fair",
        "mine", "bark", "club", "court", "date", "match", "palm", "pitch",
        "pound", "ring", "scale", "seal", "sink", "spell", "star", "tie",
        "watch", "wave", "well", "yard", "novel", "mint",
        # 2026-06-11 relationship-mining additions
        "reasonable", "bill",
        # 2026-06-11 deep-mining round 2 (classic two-sense traps)
        "band", "diamond", "lemon", "pop", "appeal", "interest", "charge",
        "draft", "suit", "check", "cabinet", "organ", "pupil", "bass",
        "crane", "bolt", "jam", "racket", "pitcher",
    ],
}

# Candidate analogies (a - b + c ~= expected). preprocess.py verifies each
# against the *slice* (nearest neighbor within these ~500 words, excluding the
# three inputs) and reports the result.
#
# Fields:
#   headline  - True for the 8 Mode-B scenarios shown by default (PRD 8.5).
#   kind      - headline | grammar | geo | pmc | bias | morphology (UI framing).
#   expected  - hoped-for answer (build-time verified). For 'bias' analogies the
#               UI presents the result descriptively, not as a "correct" answer.
#   label     - short scenario label for the UI.
#
# The two PMC headline analogies were chosen after build-time verification:
# the PRD's first-draft PMC analogies (measure-quarterly+annual,
# milestone-project+program) resolve only weakly (~0.43), so the crisp,
# high-confidence DOL analogies below were promoted to the headline slots per
# the PRD's explicit "substitute if it doesn't resolve" instruction. The
# originals are retained in the pool for open exploration.
ANALOGY_CANDIDATES = [
    # --- the 8 headline scenarios (PRD 8.5 rows 8-15) ---
    {"id": "king",        "headline": True, "kind": "headline",   "label": "king - man + woman",        "a": "king",       "b": "man",       "c": "woman",      "expected": "queen",   "teaches": "The headline analogy. The visceral 'get it' moment."},
    {"id": "capital",     "headline": True, "kind": "geo",        "label": "paris - france + japan",     "a": "paris",      "b": "france",    "c": "japan",      "expected": "tokyo",   "teaches": "Analogy mechanism with a non-loaded example; reinforces that this is computational."},
    {"id": "grammar",     "headline": True, "kind": "grammar",    "label": "bigger - big + small",       "a": "bigger",     "b": "big",       "c": "small",      "expected": "smaller", "teaches": "Geometry encodes grammar, not just meaning."},
    {"id": "pmc_worker",  "headline": True, "kind": "pmc",        "label": "worker - labor + employer",  "a": "worker",     "b": "labor",     "c": "employer",   "expected": "employee","teaches": "PMC analogy payoff - the same machinery applied to DOL vocabulary. Labor-side word for a person, shifted to the employer frame."},
    {"id": "pmc_eval",    "headline": True, "kind": "pmc",        "label": "evaluation - evaluate + monitor","a": "evaluation","b": "evaluate","c": "monitor",    "expected": "monitoring","teaches": "Second PMC analogy: the geometry encodes the grammar of PM vocabulary itself (evaluate:evaluation :: monitor:monitoring)."},
    {"id": "bias_doctor", "headline": True, "kind": "bias",       "label": "doctor - man + woman",       "a": "doctor",     "b": "man",       "c": "woman",      "expected": None,      "teaches": "The bias demo (PRD section 7) — now with the lexicon including 'physician', the math lands in a dead heat between the gender-neutral synonym and the stereotyped role. Read the numbers; that closeness IS the lesson."},
    {"id": "bias_ceo",    "headline": True, "kind": "bias",       "label": "ceo - man + woman",          "a": "ceo",        "b": "man",       "c": "woman",      "expected": "executive","teaches": "Second bias scenario. The gender shift demotes the chief title to the generic one. One result is anecdote; two from different roles is data."},
    {"id": "morphology",  "headline": True, "kind": "morphology", "label": "walking - walk + run",       "a": "walking",    "b": "walk",      "c": "run",        "expected": "running", "teaches": "Verb conjugation as geometry; lighter morphology scenario."},

    # --- PMC pool: PRD originals + verified extras (open exploration) ---
    {"id": "pmc_pay",     "headline": False, "kind": "pmc", "label": "salary - employee + employer",     "a": "salary",    "b": "employee",  "c": "employer", "expected": "wage", "teaches": "Employee-side vs employer-side of pay."},
    {"id": "bias_prog",   "headline": False, "kind": "bias", "label": "programmer - man + woman",        "a": "programmer","b": "man",       "c": "woman",    "expected": "designer", "teaches": "Weaker bias example; kept for exploration."},
    {"id": "bias_sci",    "headline": False, "kind": "bias", "label": "scientist - man + woman",         "a": "scientist", "b": "man",       "c": "woman",    "expected": "researcher", "teaches": "Title softening under gender shift; third bias data point."},
    {"id": "pmc_measure", "headline": False, "kind": "pmc", "label": "measure - quarterly + annual",   "a": "measure",   "b": "quarterly", "c": "annual",  "expected": None, "teaches": "PRD's original PMC analogy; resolves weakly but stays in-domain."},
    {"id": "pmc_plan",    "headline": False, "kind": "pmc", "label": "milestone - project + program",  "a": "milestone", "b": "project",   "c": "program", "expected": None, "teaches": "PRD's original planning analogy; resolves weakly but in-domain."},
    {"id": "pmc_mgmt",    "headline": False, "kind": "pmc", "label": "management - manager + worker",   "a": "management","b": "manager",   "c": "worker",  "expected": "employee", "teaches": "Management vs labor encoded."},
    {"id": "pmc_budget",  "headline": False, "kind": "pmc", "label": "budget - fiscal + annual",        "a": "budget",    "b": "fiscal",    "c": "annual",  "expected": "monthly",  "teaches": "Budget cadence."},
    {"id": "pmc_fund",    "headline": False, "kind": "pmc", "label": "funding - grant + budget",        "a": "funding",   "b": "grant",     "c": "budget",  "expected": "fiscal",   "teaches": "Funding/fiscal relation."},

    # --- extra verified-pool candidates (kinship/geography are most reliable) ---
    {"id": "queen_king",  "headline": False, "kind": "headline", "label": "queen - woman + man",  "a": "queen",  "b": "woman",   "c": "man",   "expected": "king",     "teaches": "Reverse of the headline."},
    {"id": "father_mom",  "headline": False, "kind": "headline", "label": "father - man + woman", "a": "father", "b": "man",     "c": "woman", "expected": "mother",   "teaches": "Kinship gender swap."},
    {"id": "uncle_aunt",  "headline": False, "kind": "headline", "label": "uncle - man + woman",  "a": "uncle",  "b": "man",     "c": "woman", "expected": "aunt",     "teaches": "Kinship gender swap."},
    {"id": "prince",      "headline": False, "kind": "headline", "label": "prince - man + woman", "a": "prince", "b": "man",     "c": "woman", "expected": "princess", "teaches": "Royalty gender swap."},
    {"id": "cap_germany", "headline": False, "kind": "geo",      "label": "berlin - germany + france", "a": "berlin", "b": "germany", "c": "france", "expected": "paris",  "teaches": "Capital analogy variant."},
    {"id": "cap_italy",   "headline": False, "kind": "geo",      "label": "rome - italy + spain",      "a": "rome",   "b": "italy",   "c": "spain",  "expected": "madrid", "teaches": "Capital analogy variant."},
    {"id": "boy_girl",    "headline": False, "kind": "headline", "label": "boy - man + woman",    "a": "boy",    "b": "man",     "c": "woman", "expected": "girl",     "teaches": "Age/gender analogy."},
]

# Deliberate "this is what failure looks like" analogies (PRD 8 Mode B). Not a
# peer scenario; the UI labels it as a teaching artifact about limitations.
# All inputs must be lexicon words (only the slice ships to the browser).
# preprocess.py evaluates each and reports the result so we can pick the one
# that most clearly looks broken.
FAILURE_CANDIDATES = [
    {"id": "fail_clusters", "a": "paris",   "b": "france", "c": "happy",
     "teaches": "Mixing unrelated clusters (geography minus country plus emotion) yields noise."},
    {"id": "fail_animal",   "a": "king",    "b": "man",    "c": "dog",
     "teaches": "A familiar analogy frame breaks when the third term is from a far-off cluster."},
    {"id": "fail_pmc",      "a": "measure", "b": "data",   "c": "love",
     "teaches": "Vocabulary arithmetic across unrelated domains produces an incoherent result."},
]
