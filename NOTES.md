# NOTES.md — working notebook for the RAG Baseline vs. Re-Ranking assignment

Decisions, observations and surprises, recorded as the work happened.

**What this file is for.** The assignment requires a report that is *not* AI-generated and that
shows the author's own understanding. So this file holds **evidence and reasoning only** —
decisions with their alternatives, measured numbers, and honest notes on what broke. It is a
lab notebook, not report prose, and nothing here is meant to be pasted into the report.

Conventions: record the alternative that was rejected and why; every number carries hardware,
repeat count and whether a warm-up was discarded; negative results stay in.

---

## 1. Decisions and why

### 1.1 Generation LLM: Gemini, via a single config variable

`GEN_MODEL_CANDIDATES = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]`, resolved once by a
smoke test to `GEN_MODEL`, which every later cell uses.

The brief names the Gemini API first and the free tier covers this workload. The candidate
list exists because model ids expire: `gemini-2.5-flash` passed its **16 Oct 2026** shutdown
date three days before this assignment was built. Hard-coding a model id in N cells is how a
notebook stops being restart-and-run-all the moment a model retires, so the id lives in one
variable and the smoke test picks the first candidate that actually answers.

Rejected: Anthropic / OpenAI (both need a funded key, and the dependent variable here is
*retrieved context*, not model strength, so a stronger generator buys nothing); local Ollama
(key-free and offline, but a multi-GB pull on deadline day, and a weaker model's own errors
would contaminate the Answer 1 vs Answer 2 comparison).

### 1.2 Temperature: originally 0, changed to the model default

**This decision was reversed mid-build, and the reversal is the interesting part.**

The original plan set `temperature=0`. The reasoning was sound for older models: §4.2 of the
brief asks us to attribute the difference between Answer 1 and Answer 2 to the retrieved
context. If the decoder samples, two runs of the *same* prompt can differ, and then any
observed difference between the two answers has two possible causes — different context, or
decoder noise. Greedy decoding removes the second cause and makes the claim falsifiable.

Checking the current `google-genai` documentation for the actual model being used overturned
this. Google's Gemini 3 guidance **strongly recommends leaving `temperature`, `top_p` and
`top_k` at their defaults** (temperature default `1.0`), because the models' reasoning
behaviour is tuned around those defaults; forcing `temperature=0` on a Gemini 3 model is a
known way to get *worse*, more repetitive output. Buying determinism at the cost of degraded
answers would damage the very comparison it was meant to protect.

So temperature is left unset, and the "only the context differs" claim is carried by two
other mechanisms instead:

1. **A byte-identical prompt template.** Both pipelines format through the same function. The
   only difference between the two prompts is which three chunks are interpolated. This is
   asserted in the notebook, not just intended.
2. **A persistent answer cache** keyed by `(model, sha256(prompt))`. An answer is generated
   once and then frozen on disk. Re-running the notebook replays the exact bytes that were
   analysed, so the committed analysis can never drift from the committed answers — which is
   a *stronger* reproducibility guarantee than `temperature=0` gives, since greedy decoding is
   still only reproducible while the served model weights stay put.

The honest limitation, which belongs in the report: with sampling enabled we cannot claim that
a *re-run* would produce these exact two answers. We can claim that these two specific answers
came from prompts differing only in their retrieved context. That is what §4.2 actually asks.

A useful side effect: when re-ranking does not change the top-3 **set**, the two prompts are
byte-identical, so the hash collides, the cache returns the same text, and Answer 1 == Answer 2
by construction — zero API calls and a provable no-op.

### 1.3 Thinking: minimised

`thinking_level="minimal"` where the installed SDK accepts it (Gemini 3.5's default moved from
`high` to `medium`; `minimal` is documented as matching "no thinking" and minimising latency).
Two reasons: §4.3 wants clean latency numbers, and a variable amount of hidden reasoning adds
variance to the one measurement we are asked to tabulate. It also keeps answers anchored to
the supplied context rather than to chains of reasoning around it.

### 1.4 LLM call budget: ≤18, with a disk cache

The free tier may allow ~20 requests/day. 6–8 queries × 2 pipelines = 12–16 generations, plus
1–2 smoke-test calls. Every answer is cached to `results/answers_cache.json` keyed by
`(model, sha256(prompt))` and **never regenerated**. Consequences: the notebook can be
re-executed any number of times at zero quota cost (required for the Phase 8 clean-kernel run),
and §4.3's LLM timings come only from uncached calls — stated explicitly rather than quietly
averaging cache hits into the number.

### 1.5 Corpus: Twilight Imperium 4th Edition, Living Rules Reference v1.3

Source (official Fantasy Flight Games CDN, HTTP 200, 3,313,611 bytes, PDF 1.6):
`https://images-cdn.fantasyflightgames.com/filer_public/da/df/dadf9f07-78f3-43ac-9cce-dd6b55b24ec2/ti4_living_rules_reference_v1_3_web.pdf`

The brief lists an in-depth board game rulebook as a suitable complex PDF. The TI4 LRR is the
strongest candidate available: it is an alphabetised, cross-referenced rules *reference* (not a
teaching manual) built almost entirely out of numbered rules, prohibitions ("cannot", "may
not"), role limits, and explicit exceptions. Those are precisely the §3 failure categories, so
the failure queries can be grounded in real text instead of contrived ones.

`data/*.pdf` is gitignored and the notebook downloads it, so the repo carries the URL and the
loader rather than redistributing the publisher's file.

### 1.6 Chunking: numbered rule boundaries, capped under 256 word-pieces

The LRR numbers nearly every rule (`61.3`, `7.2a`), which gives a semantically meaningful split
point that no generic character-window splitter would find. Each chunk carries `rule_no` and
`page`, so every retrieved chunk is citable in the report as "Rule 61.3, p.44".

The hard constraint: `all-MiniLM-L6-v2` truncates input at **256 word-pieces**. A longer chunk
is silently cut, so its tail would be invisible to retrieval while still being visible to the
LLM in the prompt. That failure looks exactly like a retrieval bug and would quietly corrupt
every §3 claim, so chunk length is **asserted** against the model's own tokenizer, not
estimated from character counts.

### 1.7 Bi-encoder: `all-MiniLM-L6-v2` — comparable size, *not* a controlled comparison

Local, free, and fast enough to re-index while iterating. It is the same rough size as the
mandated cross-encoder (both 6-layer MiniLM, ~22M parameters), which avoids confounding the
§4.1 architecture discussion with a 10× parameter gap.

**It is not a clean architecture-only comparison, and the report should not claim it is.** The
two models differ in training data and objective as well as in architecture:
`all-MiniLM-L6-v2` is trained for general-purpose sentence similarity over ~1B mixed pairs,
while `ms-marco-MiniLM-L-6-v2` is trained specifically for query→passage *relevance ranking*
on MS MARCO. Part of any observed gain therefore comes from task-matched supervision, not
purely from joint self-attention. Size is controlled; training is not.

### 1.8 Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Named in the brief. Outputs an unnormalised relevance logit per `(query, passage)` pair.
Scores are comparable *within* one query's candidate list — all that re-sorting needs — but
not across queries, so no absolute score threshold is used anywhere.

### 1.9 Vector store: FAISS `IndexFlatIP` over L2-normalised vectors

§2.b allows "cosine similarity or inner product". Inner product over L2-normalised vectors *is*
cosine similarity, so one index satisfies both readings and the notebook says so.

`IndexFlatIP` is **exact** brute-force search, chosen deliberately over IVF/HNSW: under an
approximate index, a gold chunk absent from the baseline top-10 could be explained by
approximation error rather than by bi-encoder semantics, which would undermine every §3
failure-mode claim. At one rulebook's scale exact search is free. It also keeps §4.4 honest —
the scaling argument has to concede that production systems use sublinear ANN, and that the
cross-encoder is the stage that *cannot* be made sublinear.

Rejected: Chroma (extra dependency and service surface for no benefit here); raw numpy matmul
(numerically identical, but the brief says "index chunks into a vector store").

### 1.10 Python 3.12, chosen for stability

The shell default on this machine is 3.14. 3.12 is the version the torch / sentence-transformers
/ faiss stack is best supported on, and on a same-day deadline the cost of debugging
bleeding-edge wheel availability is not worth paying. The project therefore gets a dedicated
`uv` venv pinned to 3.12; the notebook must be run on that kernel and the README says so.

### 1.11 One notebook, no `src/` package

The brief strongly prefers a single well-commented notebook. Factoring the pipeline into
importable modules would read better as software but would move the graded content out of the
graded artifact. The notebook writes machine-readable evidence to `results/*.csv` so every
table in the report traces back to a file rather than to a screenshot.

### 1.12 CSV, not parquet

Readable in a GitHub diff, openable anywhere, no engine dependency. The datasets here are
hundreds of rows.

---

## 2. Environment as measured

- Apple Silicon (`arm64`), macOS Darwin 25.4.0.
- Interpreters present: `python3.11`, `python3.12`, `python3.14` (Homebrew); `uv` available.
- Before setup: no LLM API key in the environment, no Ollama, no torch, no sentence-transformers.
- `GEMINI_API_KEY` supplied by the user into a gitignored `.env`, loaded with `python-dotenv`,
  never echoed — the notebook is committed with its outputs intact.

---

## 3. Observations

### 3.1 PDF extraction: three things that would have silently corrupted the corpus

**Column order is not reading order.** PyMuPDF's block order returned the *right-hand* column
before the left on several pages (page 7 came back as sections 3, 4, 5, then 2). Concatenating
blocks in the returned order would splice unrelated rules into the same chunk, and the splice
would be invisible in the chunk text. Blocks are therefore re-sorted column-major by
`(column, y)`, with the column decided by the block's left edge against a 306pt split.

**Card illustrations are interleaved with rule prose.** The page images carry their own text
layer, which comes back garbled and duplicated — e.g. `'NFILTRA\nNFILTRATE\nI'`, and whole
summary diagrams repeated twice on page 11. A geometric filter removes them: body prose sits at
one of four left margins (36 / 54 pt left column, 315 / 333 pt right column) or is at least
120pt wide. Over pages 6–31 that keeps **1,303 blocks and drops 105**; every dropped block was
checked by hand and all 105 are card art, flavour text, duplicated diagrams, or page numbers.

**The rule number is separated from its text by a tab, and nothing else.** The first version of
the normaliser collapsed all whitespace, which destroyed the tab and left the parser unable to
recognise a single rule boundary — it produced 0 units and 1,214 orphan blocks. Tabs are now
protected through normalisation. Worth noting because the failure was silent in the sense that
the text looked perfectly fine; only the structure was gone.

Final parse: **532 rule units across all 89 sections**, 89 `RELATED TOPICS:` lines dropped, and
**2 orphan blocks** (the glossary preamble ahead of Rule 1).

### 3.2 The document repeats itself, independently of chunk overlap

Three numeric/prohibition rules are stated twice, in different sections and different wording:

| Rule | Also stated at | Content |
|---|---|---|
| 54.3 (PDS) | 74.5 (STRUCTURES) | maximum two PDS per planet |
| 68.3 (SPACE DOCK) | 74.4 (STRUCTURES) | maximum one space dock per planet |
| 58.6 (PRODUCING UNITS) | 14.1 (BLOCKADED) | cannot produce *ships* where another player has ships, ground forces still allowed |

This matters for two reasons. It means "the gold chunk" is sometimes a **set** of chunks, so
gold rank is defined as the best-ranked member. And it gives a clean way to separate the two
causes of near-duplicate candidates: duplicates that *share* rule numbers come from the
one-unit chunk overlap, while duplicates with no shared rule numbers come from the document
stating the same rule twice. `results/near_duplicates.csv` labels each pair accordingly.

### 3.3 The SDK did not match its own documentation

The published Gemini 3.5 Flash guide shows generation as
`client.interactions.create(model=..., input=..., generation_config={"thinking_level": "minimal"})`.
The installed SDK (`google-genai` **2.31.0**) does expose `client.interactions`, but
`interactions.create` takes a request object — its parameters are
`(request, api_version, extra_headers, extra_query, extra_body, timeout, body)` — so the
documented keyword form raises `TypeError`. Generation therefore goes through
`client.models.generate_content(model, contents, config)` with
`types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"))`;
`ThinkingLevel.MINIMAL` does exist in this version. Verified by introspecting the installed
package rather than trusting the doc snippet.

Also confirmed by introspection: `types.GenerateContentConfig().temperature` is `None`, so
leaving temperature unset genuinely sends no temperature field and the provider default
applies — which is what §1.2 above relies on.

One more live detail: the docs state that `gemini-3.5-flash` is itself deprecated and
automatically routed to `gemini-3.6-flash`. It is kept as the first candidate because the
requirement was "use whichever responds", and a routed id still responds; the notebook prints
the id it resolved.

### 3.4 Key and model availability, checked without spending quota

The API key was validated with `client.models.list()` rather than a generation call, so the
check cost nothing against the ~20/day free-tier budget. The key is accepted and **63 models**
are visible, including both configured candidates: `gemini-3.5-flash` **and**
`gemini-3.1-flash-lite`. The first candidate therefore resolves and the fallback never fires.

Worth recording for context: `gemini-3.6-flash`, `gemini-3.7-flash` and `gemini-3.8-flash` are
also available on this key. `gemini-3.5-flash` is kept as the primary because it was the
specified choice and it responds; the documented auto-routing from 3.5 to 3.6 means the served
model may be newer than the id requested, which is a reason to print the resolved id rather
than assume it.

### 3.5 Dependency install: the deadline forced two pins

Installing the stack took three failed attempts and ~45 minutes of wasted download. `uv` was
re-downloading the two largest wheels from scratch on every retry rather than resuming them
(the cache grew by 244 MB, then 137 MB, while `torch` and `pymupdf` never completed), so on a
same-day deadline the fix was to fetch those two wheels directly with `curl -C -`, which does
resume, and install from local files.

Two versions were pinned to cut the download in half:

| Package | Pinned | Latest at the time | Why |
|---|---|---|---|
| `torch` | **2.5.1** (64 MB) | 2.14.1 (127 MB) | Half the bytes. Only used for forward passes on two 6-layer MiniLM models; MPS support has been present since 1.12, so nothing in this notebook behaves differently |
| `pymupdf` | **1.27.2** (23 MB) | 2.0 (67 MB) | A third of the bytes, *and* it is the exact version the extraction code was developed and verified against |

`jupyterlab` was dropped from the install entirely: executing the notebook needs only
`ipykernel` and `nbconvert`, and pulling a full Lab server was costing download time for
nothing. The README gives it as a separate optional install.

This is a reproducibility trade-off worth being explicit about: the pins are driven by download
size under deadline pressure, not by any measured incompatibility with the newer versions.

### 3.6 Quota behaved in two different ways, and both mattered

The free tier produced **two distinct failures that both arrive as HTTP 429**, and treating them
the same way cost a stalled run:

1. **Per-minute rate limit.** The first run tripped a 429 after six requests fired in quick
   succession, and recovered within minutes — both models answered again on the next probe.
   The right response is backoff and retry.
2. **Spent allowance.** Later, `gemini-3.5-flash` returned 429 with "exceeded your current
   quota" and stayed that way, while `gemini-3.1-flash-lite` answered normally. No amount of
   waiting fixes this inside one run.

The first implementation retried any 429 with escalating backoff, so when the allowance was
genuinely spent it waited ~20 minutes and got nowhere. The code now separates the two: a
rate-limited call is retried with a capped delay, and a call reporting an exhausted quota
immediately marks that model dead for the run and **fails over to the next candidate**, which
is what the brief asks for ("use whichever responds") — mid-run exhaustion is just that
situation arriving late.

A third mode showed up too: a **503 "high demand"**, which is shared-capacity pressure and
nothing to do with quota. The first retry filter only matched 429 and let the 503 kill the run.
Transient 5xx is now retried as well.

**The subtle bug the fail-over introduced.** With fail-over in place, the run then died on an
assertion — `Q9: Answer 1 from gemini-3.5-flash, Answer 2 from gemini-3.1-flash-lite`. Answer 1
had taken the last of the 3.5-flash allowance, and Answer 2 had failed over. Those two answers
differ by *both* context and model, so comparing them would measure the wrong thing entirely.
The assertion existed precisely to catch this, and it worked. The fix generates each query's
pair atomically: if the model changes between the two calls, Answer 1 is regenerated on the
surviving model. Models may still differ *across* queries — 8 queries on `gemini-3.5-flash`,
Q9 and Q10 on `gemini-3.1-flash-lite` — and `results/answers.md` and `query_summary.csv` record
which model answered each query rather than leaving it implicit.

**Honest accounting of the budget.** `LLM_CALL_BUDGET` caps calls *per notebook run*, and the
final clean run used **4** (1 smoke test + 3 generations) against 17 cache hits. A run from an
empty cache would use 21. But across the whole session — the first 8-query run, the two aborted
runs, and the model probes — roughly **30 requests** were actually spent, which is more than the
18-call ceiling that was set at the start. The ceiling governed each run, not the debugging that
got the run working, and saying otherwise would be a dodge. The on-disk cache is what kept this
from being far worse: every one of the first 16 answers was reused unchanged.

### 3.7 Chunking, as measured

172 chunks from 532 rule units across 89 sections. Token lengths by the bi-encoder's own
tokenizer: **min 20, median 179, max 229**, against a hard truncation point of 256. The
assertion that no chunk reaches 256 passes, so nothing in the corpus is silently invisible to
retrieval. The model reports `max_seq_length: 256` and a 384-dimensional embedding.

### 3.8 Rank shifts (the §4.2 evidence)

| qid | category | gold rules | baseline → re-ranked | outcome |
|---|---|---|---|---|
| Q1 | keyword_trap | 2.4 | #1 → #1 | context changed, verdict same |
| Q2 | synonym_gap | 75.0, 75.1 | #1 → #1 | context changed, verdict same |
| Q3 | exception_corner | 58.6, 14.0, 14.1 | #4 → **#6** | rank regression, both outside Top 3 |
| **Q4** | negation_constraint | 5.2 | **#6 → #1** | **promotion** |
| Q5 | role_limit | 54.3, 74.5 | #1 → #1 | context changed, verdict same |
| Q6 | exception_corner | 46.1 | never in Top 10 | **mislabelled gold — see 3.10** |
| Q7 | negation_constraint | 3.2, 3.4 | #1 → #1 | no-op, Top 3 identical |
| Q8 | buried_vital_chunk | 76.2 | #2 → #2 | context changed, verdict same |
| **Q9** | role_limit | 83.1 | **#8 → #1** | **promotion** |
| **Q10** | negation_constraint | 14.0, 14.1 | **#7 → #1** | **promotion** |

Three clean promotions, one rank regression, one no-op, and five cases where the Top 3 changed
without moving the gold chunk in or out. Two of ten answer pairs are byte-identical.

**The first attempt found only one promotion.** Eight hand-written queries produced a single
promotion (Q4), because five of them had the gold chunk at rank 1 or 2 already — the bi-encoder
simply did not fail on them. Rather than reword those until they looked like wins, the search
was made explicit and free: §4b screens 16 candidate questions through retrieval only, which
costs no quota, identifying gold by the authoritative *sentence* rather than by a remembered
rule number. That sweep found **5 promotions**, and Q9 and Q10 were adopted from it. The full
table stays in `results/failure_hunt.csv`, including the eight candidates where re-ranking
changed nothing.

### 3.9 What the promotions did to the answers

- **Q9 is the textbook case.** The gold rule sat at rank 8, and Answer 1 was a flat refusal:
  *"The provided rules do not cover this."* Answer 2, with Rule 83.1 promoted to rank 1, answers
  correctly — one transaction per neighbour per turn. A pure retrieval omission, visible as an
  omission only because the prompt forbids outside knowledge.
- **Q4 flipped its verdict.** Answer 1 said *"No, you cannot perform a tactical action on a
  system that already contains one of your command tokens"*, reasoning from the TACTICAL ACTION
  chunks (78.1, 78.2) that the baseline retrieved instead of the gold. Answer 2, with Rule 5.2
  at rank 1, says *"Yes, you can... provided it does not currently contain one of your command
  tokens"*, and adds the distinction the baseline missed entirely: other players' command tokens
  do not block activation. Same question, opposite answer, caused only by which chunk was in
  context.
- **Q10 went from diffuse to precise.** Answer 1 assembled a scattershot reply from 58.5, 58.6,
  65.2, 34.1 and 34.3 and never once used the word *blockaded* — the actual mechanic. Answer 2
  names Rule 14.1 directly. Not a hallucination, but an answer that circles the rule instead of
  stating it.

### 3.10 Two findings that weaken a naive reading of the results

**Gold labelling by rule number can understate retrieval.** Q6 looks like the worst case in the
table — "gold never entered Top 10" — but both answers are substantively correct and both cite
Rule 26.2. The reason: the custodians-token cost is stated in **two** places, and the label
picked the weaker one. Rule 46.1 (MECATOL REX) merely mentions that the token prevents landing;
Rule 26.2 (CUSTODIANS TOKEN, chunk 49) is the authoritative statement of the six-influence cost,
and retrieval put it at rank 1–2 in both pipelines. So Q6 is a labelling artifact, not a
retrieval failure — a direct consequence of the document repeating itself (3.2). Any metric
built on single-rule gold labels would have scored this as a miss.

**"Regression" needs two definitions.** Q3 moved its gold from #4 to #6 and H4 from #5 to #10:
the cross-encoder actively pushed the gold chunk *down*. Neither counts as a Top-3 regression,
because the chunk was outside the Top 3 both before and after, so the answer is unaffected. The
distinction worth keeping is between *rank* movement and *context membership* movement — only
the latter changes what the LLM sees.

### 3.11 Latency, measured on this machine

`arm64 / Darwin 25.4.0 / torch 2.5.1 / device=mps`, 172 chunks, 10 queries, 10 timed repeats
per query after a discarded warm-up (n=100 per stage).

| stage | mean ± std (ms) | median | min–max |
|---|---|---|---|
| 1. query embedding (bi-encoder) | 15.71 ± 0.96 | 15.50 | 14.61–20.58 |
| 2. FAISS search (top 10 of 172) | **0.02 ± 0.01** | 0.01 | 0.01–0.06 |
| 3. cross-encoder scoring (10 pairs) | 151.54 ± 20.66 | 147.04 | 107.73–244.19 |
| **Pipeline 1 retrieval total** | **16.11 ± 1.30** | 15.71 | 14.81–21.71 |
| **Pipeline 2 retrieval total** | **169.72 ± 23.62** | 161.78 | 125.95–252.65 |
| LLM generation (uncached only, n=3) | 8603 ± 4551 | 8625 | 4042–13143 |

Re-ranking adds **153.6 ms, a 10.54× increase** in retrieval time for ten candidates. Three
things are worth drawing out of this table rather than leaving implicit:

- **The vector search is not the cost.** FAISS exact search over 172 chunks takes 0.02 ms —
  roughly 800× less than embedding the query. At this scale the retrieval bill is dominated by
  one forward pass through the bi-encoder.
- **Re-ranking's 10× is on a tiny base.** 154 ms is a real cost, but the LLM call that follows
  averages **8.6 seconds**, so re-ranking is about **1.8%** of end-to-end latency. Stated as a
  multiple it looks alarming; stated against the pipeline it is close to free.
- **The LLM figure is the least trustworthy number here.** n=3, std ±4.5s (over 50% relative),
  and it comes from two different models — so it establishes an order of magnitude and nothing
  finer. Cached answers are excluded rather than averaged in, which would have divided the
  apparent cost by the hit rate.

### 3.12 Scaling (the §4.4 evidence)

Measured cross-encoder throughput: **71.0 (query, chunk) pairs/sec** on this device.

| corpus chunks | cross-encode entire corpus, per query | baseline retrieval | re-rank Top 10 | ratio |
|---|---|---|---|---|
| 172 (this corpus) | 2.42 s | 0.016 s | 0.170 s | 14× |
| 10,000 | 141 s | 0.016 s | 0.170 s | 830× |
| 1,000,000 | 14,090 s (**3.9 hours**) | 0.016 s | 0.170 s | 83,018× |

The mechanism behind the numbers: the bi-encoder embeds each chunk **once, at index time**, and
a query then costs one forward pass plus a sublinear index lookup. The cross-encoder cannot
reuse anything, because its score is a function of query and chunk *jointly* — there is no
per-chunk vector to precompute, so every query requires N forward passes. That is O(N) per
query with a large constant, and it is why the two-stage design exists: it confines the O(N)
stage to N=10. Even on this toy corpus, exhaustive cross-encoding is already 14× slower than
the entire re-ranking pipeline, and the gap widens linearly forever.

### 3.13 Near-duplicate candidates: the cost of the overlap decision

7 near-duplicate pairs (cosine ≥ 0.90) appeared in the Top 10 across the 10 queries, and **all 7
share rule numbers** — every one is caused by the deliberate one-rule chunk overlap, not by the
document repeating itself. (The document *does* repeat itself, per 3.2, but those twins never
co-occurred in a Top 10.)

Two of them are real damage rather than noise: in **Q7 and Q9**, chunks 8 and 9 (both containing
Rule 3.3) were *both inside the baseline Top 3*, so two of the three context slots carried
nearly the same text and the prompt effectively had two useful excerpts instead of three. In Q9
that is part of why the baseline refused to answer. Honest reading: `OVERLAP_UNITS = 1` buys
boundary safety and pays for it in candidate-slot dilution, and at `TOP_K_CONTEXT = 3` that
price is high. A deduplication pass before truncating to the Top 3 would be the obvious fix —
and is deliberately *not* implemented here, because it is not part of what the assignment asked
for.
