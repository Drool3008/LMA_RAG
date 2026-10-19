# Cell guide — `notebooks/rag_baseline_vs_reranking.ipynb`

The notebook keeps short comments so it reads quickly. The long reasoning that used to sit in
docstrings and comment banners lives here instead, cell by cell, with nothing dropped.

Each entry gives: what the cell does (and which variables later cells depend on), why it is
built that way, which assignment section it serves, and any gotcha. Quoted blocks marked
*preserved* are the verbatim text that was moved out of the notebook.

Assignment: `../RAG_LMA_Assignment_v1.pdf`. Decisions and measured results: `../NOTES.md`.

| cell | type | what it is | PDF section |
|---|---|---|---|
| 0 | md | title and pipeline summary | — |
| 1 | code | configuration | §1 |
| 2 | code | corpus download | §1 |
| 3 | code | model loading, `count_tokens` | §1 |
| 4 | code | Gemini client, retry/fail-over, smoke test | §1 |
| 5 | md | chunking rationale | §1 |
| 6 | code | PDF extraction and rule parsing | §1 |
| 7 | code | chunk packing, token cap assert | §1 |
| 8 | md | Pipeline 1 heading | §2 P1 |
| 9 | code | embed, FAISS index, `baseline_retrieve` | §2 P1.a–b |
| 10 | code | prompt template, answer cache, `generate` | §2 P1.c, P2.d |
| 11 | md | Pipeline 2 heading | §2 P2 |
| 12 | code | `rerank` | §2 P2.b–c |
| 13 | md | query set heading | §3 |
| 14 | code | the 10 queries, gold resolution | §3 |
| 15 | md | failure-hunt heading | §3 |
| 16 | code | 16-candidate retrieval-only hunt | §3 |
| 17 | md | run-loop heading | §2 |
| 18 | code | both pipelines over every query | §2, §4.2 |
| 19 | md | rank-shift heading | §4.2 |
| 20 | code | rank-shift table | §4.2 |
| 21 | md | near-duplicate heading | §3 |
| 22 | code | near-duplicate log | §3 |
| 23 | md | answer-comparison heading | §4.2 |
| 24 | code | `answers.md` writer | §4.2 |
| 25 | md | latency heading | §4.3 |
| 26 | code | per-stage timing | §4.3 |
| 27 | md | scaling heading | §4.4 |
| 28 | code | CE throughput and extrapolation | §4.4 |
| 29 | md | outputs index | — |

---

## Cell 0 — Title and pipeline summary

**What** Markdown only. States the two pipelines and points at `results/` and `NOTES.md`.

**PDF** Framing for §2.

> *Preserved (original markdown):*
>
> # Baseline Retrieval vs. Cross-Encoder Re-Ranking
>
> Two retrieval pipelines over **one** chunked corpus — the *Twilight Imperium 4th Edition
> Living Rules Reference* (v1.3, official FFG PDF).
>
> * **Pipeline 1 (baseline)** — bi-encoder embeddings in a FAISS index → **Top 10** by cosine
>   similarity → **Top 3** straight into the prompt → **Answer 1**.
> * **Pipeline 2 (re-ranking)** — the **same Top 10** → all 10 `(query, chunk)` pairs scored by
>   a cross-encoder → re-sorted → **Top 3** → the **identical prompt template** → **Answer 2**.
>
> Everything downstream (rank shifts, answer comparison, latency, scaling) is written to
> `results/` as CSV so the report can cite a file rather than a screenshot.
> Design decisions and their justifications live in `NOTES.md`.

## Cell 1 — Configuration

**What** Every tunable in one place. Creates `ROOT`, `DATA`, `RESULTS`, `PDF_URL`, `PDF_PATH`,
`BI_ENCODER_NAME`, `CROSS_ENCODER_NAME`, `GEN_MODEL_CANDIDATES`, `GEN_MODEL` (None until the
smoke test in cell 4 sets it), `TOP_K_RETRIEVE=10`, `TOP_K_CONTEXT=3`, `MAX_CHUNK_TOKENS=230`,
`TOKENIZER_HARD_CAP=256`, `OVERLAP_UNITS=1`, `LLM_CALL_BUDGET=22`, `CACHE_PATH`,
`NEAR_DUP_COSINE=0.90`, `LATENCY_REPEATS=10`. Nearly every later cell reads these.

**Why** `ROOT` is resolved from the working directory so the notebook runs both from the repo
root and from `notebooks/`. The generation model is a *candidate list* because model ids expire:
`gemini-2.5-flash` passed its 2026-10-16 shutdown date during this project, so the smoke test
resolves whichever candidate answers rather than hard-coding one id in many cells.

**PDF** §1 (setup), and fixes the §2 sizes Top 10 / Top 3.

**Gotcha** `GEN_MODEL` is deliberately `None` here. Anything that calls the LLM before cell 4
would fail loudly rather than silently using a default.

> *Preserved (configuration banner):*
>
> ```
> # =============================================================================================
> # Configuration. Every knob in the notebook lives here.
> # =============================================================================================
> ```
>
> *Preserved (model choice):*
>
> ```
> # Models. Bi-encoder and cross-encoder are both 6-layer MiniLM, i.e. comparable size — but NOT
> # a controlled architecture-only comparison, since they differ in training data and objective
> # as well (see NOTES.md 1.7).
> ```
>
> *Preserved (LLM budget):*
>
> ```
> # LLM budget. Every answer is cached on disk and never regenerated, so re-running this notebook
> # costs zero API calls. Raised from 18 to 22 after the first run: eight hand-written queries
> # yielded only ONE clean promotion (five had the gold chunk at rank 1-2 already, so re-ranking
> # had nothing to repair), and the assignment requires at least two. Q9 and Q10 were found by the
> # retrieval-only hunt in §4b, which costs no quota, and adding their two answers each is what
> # took the ceiling to 22. The observed limit is per-minute, not per-day — see NOTES.md 3.6.
> ```

## Cell 2 — Corpus download

**What** Fetches `PDF_URL` to `PDF_PATH` if absent, prints the byte count.

**Why** `data/*.pdf` is gitignored, so the repo carries the URL and the loader instead of
redistributing the publisher's file. A fresh clone therefore reproduces the corpus itself.

**PDF** §1 (a single complex PDF as the dataset).

## Cell 3 — Model loading and `count_tokens`

**What** Imports torch and sentence-transformers, picks `DEVICE` (`mps` when available, else
`cpu`), loads `bi_encoder` and `cross_encoder`, builds the `HARDWARE` string used by the latency
and scaling tables, and defines `count_tokens`. Cells 7, 9, 12, 26 and 28 all depend on these.

**Why** `HARDWARE` is captured here so the §4.3 table carries the machine it was measured on —
these are wall-clock numbers from one laptop, not published benchmarks.

**PDF** §1 (embedding model), and supplies the device string §4.3 needs.

**Gotcha — the 256 word-piece cap.** This is the single most important constraint in the
notebook; see cell 7, which asserts against it.

> *Preserved (`count_tokens` docstring):*
>
> Word-piece length according to the bi-encoder's OWN tokenizer, special tokens included.
> Character or word heuristics are not good enough here: the model truncates at 256
> word-pieces, and anything past that is invisible to retrieval while still visible to the
> LLM in the prompt — a bug that would present as a retrieval failure.

## Cell 4 — Generation client, retry/fail-over, smoke test

**What** Loads `.env`, builds the `genai.Client`, defines `THINKING_CFG` / `PLAIN_CFG`,
the counters `LLM_CALLS` / `_exhausted` / `GEN_CONFIG`, the pacing constants
`RPM_PAUSE_S` / `MAX_RETRY` / `MAX_WAIT_S` / `TRANSIENT`, the helpers `_retry_delay`,
`_is_quota_exhaustion`, `_raw_call`, `advance_model`, then runs the smoke test that assigns
`GEN_MODEL`. Cell 10's `generate` is the only caller of `_raw_call` in normal operation.

**Why** Temperature is left unset on purpose, and the reproducibility claim rests on the prompt
template plus the on-disk cache instead. Thinking is pinned to `MINIMAL` so hidden reasoning
does not add variance to the §4.3 numbers.

**PDF** §1 (the LLM API), and protects the §4.2 comparison.

**Gotchas**
- Only *successful* generations count against `LLM_CALL_BUDGET`; a rejected request is not a
  generation.
- The duration returned by `_raw_call` covers the successful request alone, so time spent
  waiting out a rate limit cannot leak into the latency table.
- A 429 is ambiguous and is split two ways: a per-minute rate limit is retried, while a spent
  allowance marks the model dead for the run and fails over.
- The key is read via `python-dotenv` and never printed, because the notebook is committed with
  its outputs.

> *Preserved (generation-client banner):*
>
> ```
> # Generation client.
> #
> # Temperature is deliberately LEFT UNSET. Google's Gemini 3 guidance strongly recommends not
> # overriding temperature/top_p/top_k because the models are tuned around their defaults, and
> # google-genai sends no temperature at all while the field is None. The "only the retrieved
> # context differs" claim is therefore carried by (a) a byte-identical prompt template, asserted
> # in §5, and (b) a persistent answer cache that freezes each generation on disk.
> #
> # Thinking is set to MINIMAL: §9 asks for latency numbers, and a variable amount of hidden
> # reasoning would add variance to the one quantity being tabulated.
> ```
>
> *Preserved (`TRANSIENT` comment):*
>
> ```
> # Transient conditions worth waiting out rather than failing the run. 429 is the per-minute
> # rate limit; 5xx/UNAVAILABLE is the shared-capacity "high demand" response, which the first
> # ten-query run hit on a 503 and which has nothing to do with quota.
> ```
>
> *Preserved (`_retry_delay` docstring):*
>
> Honour the server's retryDelay when it sends one, otherwise back off linearly. Capped:
> waiting ten minutes on a model whose daily quota is gone is not patience, it is a hang.
>
> *Preserved (`_is_quota_exhaustion` docstring):*
>
> A 429 means two different things. "exceeded your current quota" is a spent allowance that
> no amount of waiting fixes within this run, so the right response is to fail over to the
> next candidate model. A plain per-minute rate limit is worth retrying.
>
> *Preserved (`_raw_call` docstring):*
>
> One generation request -> (text, request_seconds).
>
> The free tier here limits requests PER MINUTE, not per day: the first run tripped a 429
> after six requests in quick succession and recovered within minutes. Transient failures
> (429 rate limits and 5xx "high demand") are retried with backoff against the SAME model —
> switching models mid-run would make Answer 1 and Answer 2 incomparable — and requests are
> spaced at least RPM_PAUSE_S apart.
>
> Two details that keep the measurements honest: only successful generations count against
> LLM_CALL_BUDGET, and the returned duration covers the successful request alone, so waiting
> out a rate limit can never leak into the latency table.
>
> *Preserved (`advance_model` docstring):*
>
> Move GEN_MODEL to the next candidate that still answers. Used when a model's quota runs
> out mid-run: the assignment says to use whichever candidate responds, and mid-run
> exhaustion is exactly that situation arriving late.

## Cell 5 — Chunking rationale

**What** Markdown only. Explains the three preprocessing decisions implemented in cell 6.

**PDF** §1 (preparing the corpus).

> *Preserved (original markdown):*
>
> ## 1. Corpus and chunking
>
> The Living Rules Reference is a two-column, alphabetised rules *reference*: almost every
> statement carries a rule number (`58.6`, `2.4`), which gives a semantically meaningful split
> point that no generic character-window splitter would find. Chunks therefore follow rule
> boundaries, and each one records its rule numbers and pages so every retrieved chunk is
> citable in the report as "Rule 58.6, p.21".
>
> Three preprocessing decisions, all visible in the code below:
>
> 1. **Pages 6–31 only** — the numbered rules. The table of contents, the setup walkthrough and
>    the back-of-book index are excluded: they are navigation, not rule content, and the index
>    in particular is a list of bare keywords plus rule numbers that would inject lexically
>    matching but semantically empty chunks, manufacturing "keyword traps" that are artifacts
>    of preprocessing rather than properties of the document.
> 2. **Geometric block filter** — PyMuPDF returns card illustrations and duplicated summary
>    diagrams interleaved with body prose, and returns the right-hand column *before* the left
>    on some pages. Blocks are kept by column margin/width and re-sorted column-major.
> 3. **`RELATED TOPICS:` lines are dropped** — they are cross-reference keyword lists. Keeping
>    them would act as an artificial lexical magnet and distort the very retrieval comparison
>    this notebook is measuring.

## Cell 6 — PDF extraction and rule parsing

**What** Defines `RULES_PAGES` (0-indexed 5–30 = printed pages 6–31), `BODY_MARGINS`,
`COL_SPLIT`, `_LEAD`, and the functions `_norm`, `body_blocks`, `parse_units`. Opens the PDF as
`doc` and produces `units` plus `parse_stats`, consumed by cell 7.

**Why** The rule number is the only structural signal in the extracted text, and it is separated
from its text by a tab — so normalisation protects tabs. Blocks are re-sorted column-major
because PyMuPDF's native order is not reading order.

**PDF** §1.

**Gotchas**
- An early version of `_norm` collapsed all whitespace, destroyed the tab, and produced 0 units
  with 1,214 orphan blocks — the text looked fine, only the structure was gone.
- Card art and duplicated summary diagrams share the page with body prose; the geometric filter
  drops 105 blocks over pages 6–31 and keeps 1,303.
- Section preambles are numbered `<section>.0` to match the rulebook's own index convention.

> *Preserved (`_norm` docstring):*
>
> Normalise one PDF block. Tabs are protected because the rule number is separated from
> its text by a tab, and collapsing it would destroy the only structural signal available.
>
> *Preserved (`body_blocks` docstring):*
>
> Body prose only, in true reading order (left column top-to-bottom, then right).
>
> *Preserved (`parse_units` docstring):*
>
> Parse the rules into units: one numbered rule plus its bullets and wrapped lines.
> A section's unnumbered preamble becomes `<section>.0`, matching the document's own index
> convention (the back index refers to section preambles as e.g. "agenda phase, 8.0").

## Cell 7 — Chunk packing and the token-cap assertion

**What** Defines `_rule_key` and `build_chunks` (with the inner `split_oversize`), builds
`chunk_rows` and the `chunks` DataFrame, asserts the token cap and the index/`chunk_id`
identity, and writes `results/chunks.csv`. `chunks` is the corpus for every later cell.

**Why** Chunks are packed along rule boundaries within a section, each prefixed with its section
title, because sub-rules are often a single clause whose subject lives in the heading.

**PDF** §1, and produces the one shared corpus §2 requires.

**Gotchas**
- `assert chunks["n_tokens"].max() < TOKENIZER_HARD_CAP` is the guard against silent truncation.
  Measured result: min 20, median 179, max 229 against the 256 cap.
- `assert (chunks.index == chunks["chunk_id"]).all()` matters because later cells address chunks
  with `.loc[chunk_id]`.
- Reading `chunks.csv` back needs `dtype=str` on the rule columns, or pandas turns `"31.10"`
  into the float `31.1` and the report misquotes the rule.

> *Preserved (`build_chunks` docstring):*
>
> Pack consecutive rule units of the same section into chunks under the token cap.
>
> Two details that matter for the experiment:
> * Each chunk is prefixed with its section title ("ABILITIES (Rule 1)"). Sub-rules are often
>   one short clause whose subject lives in the heading, so without the prefix the embedding
>   of "9.1 An anomaly is identified by a red border" carries no topical anchor at all.
> * `overlap=1` repeats the last rule of a chunk as the first rule of the next, so a rule and
>   its immediate context stay together across a boundary. The cost is near-duplicate
>   candidates, which §4 logs explicitly rather than hiding.
>
> *Preserved (`split_oversize` docstring):*
>
> One rule longer than the cap is divided at sentence boundaries, keeping its rule
> number so gold-chunk matching still works.

