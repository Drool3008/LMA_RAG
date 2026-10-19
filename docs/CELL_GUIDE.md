# Cell guide — `notebooks/rag_baseline_vs_reranking.ipynb`

The notebook keeps short comments so it reads quickly. The long reasoning that used to sit in
docstrings and comment banners lives here instead, cell by cell, with nothing dropped.

Each entry gives: what the cell does (and which variables later cells depend on), why it is
built that way, which assignment section it serves, and any gotcha. Quoted blocks marked
*preserved* are the verbatim text that was moved out of the notebook.

Assignment: `../RAG_LMA_Assignment_v1.pdf`.

Some preserved blocks below point at `NOTES.md` section numbers. That is the author's working
decision log and is deliberately not in this repo — the report has to be in the author's own
words. Those pointers are kept as they were written; this guide stands on its own without them.

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

**What** Markdown only. States the two pipelines, points at `results/`, and carries the
`## 0. Setup` heading for cells 1-4.

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
`TOKENIZER_HARD_CAP=256`, `OVERLAP_UNITS=1`, `LLM_CALL_BUDGET=26`, `CACHE_PATH`,
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
- The smoke test is **not** fatal. If no candidate answers (quota spent, or the service is down)
  it warns and continues, because every answer is in the committed cache and a run that needs no
  new generation must still work. `generate` raises only when an uncached answer is really needed.
- A model that exhausts its transient retries is added to `_exhausted` too, so later queries fail
  fast instead of re-waiting the same backoff.
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

## Cell 8 — Pipeline 1 heading

**What** Markdown only.

**PDF** §2 Pipeline 1.

> *Preserved (original markdown):*
>
> ## 2. Pipeline 1 — baseline retrieval (bi-encoder)
>
> Chunks are embedded independently, L2-normalised, and stored in a FAISS `IndexFlatIP`. Inner
> product over L2-normalised vectors *is* cosine similarity, which satisfies the assignment's
> "cosine similarity or inner product" either way.
>
> `IndexFlatIP` is **exact** brute-force search, chosen over an approximate index (IVF/HNSW) on
> purpose: under an ANN index a gold chunk missing from the Top 10 could be blamed on
> approximation error instead of on bi-encoder semantics, which would make every failure-mode
> claim below unfalsifiable.

## Cell 9 — Embeddings, FAISS index, `baseline_retrieve`

**What** Builds `corpus_texts`, `corpus_emb` (L2-normalised), the FAISS `index`, and
`baseline_retrieve(query, k=TOP_K_RETRIEVE)` returning 10 dicts with `chunk_id`, `bi_score`,
`bi_rank`. `corpus_emb` is reused by the near-duplicate log in cell 22; `baseline_retrieve` is
called by cells 16, 18 and 26.

**Why** Inner product over normalised vectors *is* cosine, so one index satisfies the
assignment's "cosine similarity or inner product" either way. `IndexFlatIP` is exact rather than
approximate on purpose.

**PDF** §2 P1.a (indexed with a dense bi-encoder) and §2 P1.b (Top 10 by cosine).

**Gotcha** The candidate list this returns is the *same* list Pipeline 2 consumes — it is the
pivot of the whole experiment, and cell 18 asserts it.

> *Preserved (`baseline_retrieve` docstring):*
>
> Pipeline 1 retrieval. Returns the ranked candidate list that BOTH pipelines consume.

## Cell 10 — Prompt template, answer cache, `generate`

**What** Defines `PROMPT_TEMPLATE`, `build_prompt`, loads `_cache` from
`results/answers_cache.json`, and defines `_cache_key`, `_cache_lookup`, `generate`. Cell 18
calls `generate` twice per query.

**Why** One template, one `build_prompt`, used by both pipelines — the only thing that differs
between Answer 1 and Answer 2 is which three chunks get interpolated. The "answer only from the
excerpts" instruction is what makes a retrieval omission *visible* as an omission.

**PDF** §2 P1.c (Top 3 → Answer 1) and §2 P2.d (identical prompt structure → Answer 2).

**Gotchas**
- Cache key is `sha256(model + "\0" + prompt)`, so identical prompts collapse onto one entry:
  when re-ranking leaves the Top 3 set unchanged, Answer 2 *is* Answer 1 by construction, at
  zero API cost.
- `_cache_lookup` searches the current model first and then the other candidates, so answers
  generated before a quota fail-over stay usable instead of being silently regenerated.
- `strict_model=True` restricts the lookup to the current model; cell 18 uses it to repair a
  pair split across two models.

> *Preserved (prompt banner):*
>
> ```
> # The prompt. ONE template, used by both pipelines — the only thing that changes between
> # Answer 1 and Answer 2 is which three chunks get interpolated, and that is asserted later.
> # "Answer only from the excerpts" matters: without it the model can paper over a missing chunk
> # from prior knowledge of the game, and an omission caused by retrieval would never surface.
> ```
>
> *Preserved (cache comment):*
>
> ```
> # Persistent answer cache keyed by (model, sha256(prompt)). An answer is generated once and
> # then frozen on disk, so the committed analysis can never drift from the committed answers and
> # re-running the notebook costs zero API quota.
> ```
>
> *Preserved (`_cache_lookup` docstring):*
>
> Look for this prompt under the current model first, then under the other candidates.
>
> This matters because a model can run out of quota mid-project: answers generated earlier on
> one model must stay usable instead of being silently regenerated on another, which would
> both burn quota and make Answer 1 and Answer 2 come from different models within one query.
> The entry records which model produced it, and the run loop asserts each query's pair agrees.
>
> *Preserved (`generate` docstring):*
>
> -> (answer, prompt, cache_hit, seconds, model). Identical prompts collapse onto one cache
> entry, so when re-ranking leaves the Top 3 set unchanged, Answer 2 IS Answer 1, by
> construction.

## Cell 11 — Pipeline 2 heading

**What** Markdown only.

**PDF** §2 Pipeline 2.

> *Preserved (original markdown):*
>
> ## 3. Pipeline 2 — cross-encoder re-ranking
>
> The cross-encoder receives the **same 10 candidates** the baseline produced — it never
> re-retrieves — scores every `(query, chunk)` pair with joint self-attention over both texts,
> and the list is re-sorted by that score. Scores are raw relevance logits, comparable *within*
> one query's candidate list (all that re-sorting needs) but not across queries, so no absolute
> threshold is used anywhere.

## Cell 12 — `rerank`

**What** Scores all `(query, chunk)` pairs for the candidate list with `cross_encoder.predict`,
sorts descending by `ce_score`, stamps `ce_rank` 1..10. Called by cells 16, 18 and 26.

**Why** It takes `candidates` as an argument and never retrieves anything itself, which is what
makes "the same Top 10" structurally true rather than merely intended.

**PDF** §2 P2.a (same candidates), P2.b (all 10 pairs through the cross-encoder), P2.c
(re-sorted descending).

**Gotcha** Cross-encoder outputs are unnormalised relevance logits — comparable *within* one
query's list, not across queries. No absolute score threshold is used anywhere for that reason.

> *Preserved (`rerank` docstring):*
>
> Pipeline 2. Consumes the baseline candidate list and returns it re-sorted.

## Cell 13 — Query set heading

**What** Markdown only.

**PDF** §3.

> *Preserved (original markdown):*
>
> ## 4. Query set
>
> Eight questions, hand-written after reading the rulebook, each with its **gold rule numbers**
> recorded before any retrieval was run. They are deliberately phrased the way a player would
> ask — paraphrases and synonyms rather than the rulebook's own section vocabulary — because a
> query that quotes the heading verbatim tests nothing.
>
> The categories are the ones named in the assignment. Whether a category actually *produces* a
> baseline failure in this document is an empirical result, reported as found: promotions,
> no-ops and regressions all stay in the table.

## Cell 14 — The queries and gold resolution

**What** Defines `QUERIES` (Q1–Q12, each with `qid`, `category`, `question`, `gold_rules`,
`why`), defines `gold_chunk_ids`, then resolves and asserts `gold_chunk_ids` for every query.
Cell 18 iterates `QUERIES`.

**Why** Gold rule numbers were written down *before* any retrieval ran, and the questions are
phrased the way a player would ask rather than quoting section headings, because a query that
repeats the heading verbatim tests nothing. Q9 and Q10 were added later from the cell 16 hunt;
their `why` fields say so.

**PDF** §3 (the failure queries and their categories).

**Gotchas**
- Gold is a *set*, not one chunk: overlap puts a rule in two adjacent chunks, and the rulebook
  states some rules twice in different sections. "Gold rank" means the best-ranked member.
- The `assert q["gold_chunk_ids"]` catches a gold rule number that matches no chunk, which would
  otherwise look like a retrieval failure.
- Q6's gold label was originally only 46.1 (MECATOL REX), which merely *mentions* the token.
  That scored as "gold never entered Top 10" while both answers were correct, because the
  authoritative statement of the six-influence cost is 26.2 (CUSTODIANS TOKEN). Gold is now both
  rules and the row reads #3 → #1. A caution worth keeping: a single-rule gold label can
  manufacture a retrieval failure that is really a labelling error.
- Q8 was relabelled from `buried_vital_chunk` to `exception_corner`: its gold sat at #2 before
  and after, so it demonstrates an exception clause, not a buried chunk. The buried-chunk
  phenomenon is a rank property, shown by the promotions (Q4, Q9, Q10, Q12).
- Q11 and Q12 were added after auditing the first ten. Q11 is the keyword trap; Q12 is an
  exceptions/corner case that the cross-encoder actually repairs (#4 → #2).

> *Preserved (`gold_chunk_ids` docstring):*
>
> Chunk is gold if it contains any gold rule number. Overlap means a rule can appear in two
> chunks, so this is a set, and 'best gold rank' means the best-ranked member of that set.

## Cell 15 — Failure-hunt heading

**What** Markdown only.

**PDF** §3.

> *Preserved (original markdown):*
>
> ## 4b. How the failure queries were found (retrieval only, no LLM calls)
>
> The first pass used eight hand-written questions and produced only **one** clean promotion:
> five of the eight had the gold chunk at rank 1 or 2 already, so re-ranking had nothing to
> repair. Rather than quietly reword those queries until they looked like wins, the search was
> made explicit and cheap: 16 candidate questions, each with its gold chunk identified by the
> **authoritative sentence** (a regex over the chunk text) instead of a remembered rule number,
> scored through both pipelines. Retrieval and re-ranking cost no API quota, so this sweep is
> free and fully reproducible.
>
> Q9 and Q10 came out of this table. Everything else stays in it, including the cases where
> re-ranking **hurt** or could not help at all — those are findings, not failures to hide.

## Cell 16 — The retrieval-only failure hunt

**What** Defines `HUNT_CANDIDATES` (36 questions, each with a regex identifying its gold
sentence), runs both pipelines over each, classifies the verdict, writes
`results/failure_hunt.csv` as `hunt_df`, and prints promotions by category plus a dedicated
keyword-trap table.

The second sweep (the `K`/`L` ids) targets keyword traps specifically: each question is phrased
in the vocabulary of a section that does **not** hold the answer. Q11 and Q12 came from here.

**Why** Gold is identified by the *authoritative sentence* rather than a remembered rule number,
which is more robust to mislabelling. The sweep costs no API quota, so searching widely is free.

**PDF** §3 — this is the documented discovery method behind Q9 and Q10.

**Gotcha** The failures stay in the table on purpose: candidates whose regex matched nothing,
candidates whose gold never entered the Top 10, the no-ops, and the rank regressions (H4 #5 → #10,
L4 #1 → #4). The headline result is the keyword-trap row count: **17 keyword traps screened, 0
promotions**.

## Cell 17 — Run-loop heading

**What** Markdown only, short.

**PDF** §2.

## Cell 18 — Both pipelines over every query

**What** The core loop. Per query: `baseline_retrieve` → `rerank` → `base_top3` / `rr_top3` →
two `generate` calls → one row per candidate in `runs` and one summary row in `summary`. Writes
`results/retrieval_runs.csv` and `results/query_summary.csv`; leaves `runs_df` and `summary_df`
for cells 20, 22, 24 and 26.

**Why** Outcomes are classified mechanically (`promotion`, `regression`, `no_op_identical`,
`no_op_reordered`, `changed_context_same_verdict`) so the §4.2 table is derived from data rather
than retyped by hand.

**PDF** §2 (both pipelines end to end) and §4.2 (the rank-shift and answer evidence).

**Gotchas — three assertions carry the experiment**
1. `assert {c["chunk_id"] for c in candidates} == {c["chunk_id"] for c in reranked}` — Pipeline 2
   re-ranks the baseline candidates and never re-retrieves.
2. `assert _skeleton(p1, base_top3) == _skeleton(p2, rr_top3)` — masking each prompt's own
   excerpts must reduce both to the same skeleton. This is what "identical prompt structure"
   means operationally.
3. `assert m1 == m2` — both answers of one query must come from the same model. This fired for
   real on Q9 and caught a genuine defect.

When the two answers disagree on model, the pair is re-aligned on whichever model already has
**both** answers cached, and regenerated only if neither does. Without that, a stale entry left
by an aborted run can win the lookup for one half of a pair and deadlock the assertion.

If no model is available and a pair is not cached, the answers are recorded as pending
(`answers_generated = False`) and the retrieval evidence for that query is still produced. That
is why Q11 and Q12 currently have rank data but no Answer 1 / Answer 2.

> *Preserved (mid-pair fail-over comment):*
>
> ```
> # A model can run out of quota BETWEEN the two calls of one query — it happened here, with
> # Answer 1 landing on gemini-3.5-flash and Answer 2 failing over to the lite model. Comparing
> # those two would measure the model, not the retrieved context, so Answer 1 is regenerated on
> # the surviving model. Models may still differ ACROSS queries; that is recorded per row.
> ```
>
> *Preserved (prompt-skeleton comment):*
>
> ```
> # The two prompts may differ ONLY in the interpolated excerpts: masking each prompt's own
> # excerpt texts must reduce both to the same skeleton. This is what "identical prompt
> # structure" means operationally, so it is checked rather than asserted in prose.
> ```

## Cell 19 — Rank-shift heading

**What** Markdown only, short.

**PDF** §4.2.

## Cell 20 — Rank-shift table

**What** Builds `shift` from `summary_df`, formats the `shift` column (`#8 -> #1`), writes
`results/rank_shifts.csv`, then prints the full candidate detail for each promotion.

**Why** The per-promotion print shows what the cross-encoder pushed *down* to make room, which
is the part a bare before/after number hides.

**PDF** §4.2 (document the rank changes).

**Gotcha** The formatter uses `pd.notna`, not truthiness, because a missing gold rank is `NaN`
and `bool(float("nan"))` is `True` — the naive version printed `#nan -> #nan`.

The table also carries `best_gold_chunk`, because the brief's example phrasing is "Chunk X moved
from #8 → #1" and a rule number alone does not identify a chunk. A separate line prints which
queries were buried at 4-10 and then promoted into the Top 3 — the one category that is a rank
property rather than a question type.

## Cell 21 — Near-duplicate heading

**What** Markdown only.

**PDF** §3 (a document/preprocessing-specific effect).

> *Preserved (original markdown):*
>
> ## 7. Near-duplicate candidates from chunk overlap
>
> Chunks overlap by one rule unit, so a rule can appear in two adjacent chunks. When both land
> in the Top 10 they consume two of the ten candidate slots (and, if both reach the Top 3, two
> of the three context slots) with nearly the same text. Logged rather than silently tolerated,
> because it is a direct cost of the overlap decision.

## Cell 22 — Near-duplicate log

**What** For each query, compares every pair of retrieved chunks by cosine over `corpus_emb`,
records pairs at or above `NEAR_DUP_COSINE`, writes `results/near_duplicates.csv`.

**Why** Pairs that *share* rule numbers come from the deliberate chunk overlap; pairs that share
none come from the document stating the same rule twice. The `cause` column separates them, so
the cost of the overlap decision is attributable rather than guessed.

**PDF** §3.

**Gotcha** Measured: 7 pairs, all from chunk overlap. In Q7 and Q9 *both* twins were inside the
baseline Top 3, so two of three context slots held near-identical text.

## Cell 23 — Answer-comparison heading

**What** Markdown only, short.

**PDF** §4.2.

## Cell 24 — `answers.md` writer

**What** Writes `results/answers.md` — per query: the question, gold rules, rank shift, outcome,
both Top 3 sets, the answering model, and Answer 1 / Answer 2 side by side. Then prints the
pairs that differ.

**Why** The answers live in a markdown file rather than a CSV column because they are prose
meant to be read side by side; `query_summary.csv` deliberately drops them.

**PDF** §4.2 (compare Answer 1 vs Answer 2).

**Gotcha** The header reports the per-query model because the project spans two models, and
states that both answers of any one query always come from the same one.

## Cell 25 — Latency heading

**What** Markdown only.

**PDF** §4.3.

> *Preserved (original markdown):*
>
> ## 9. Latency profiling
>
> Each stage is timed on its own, `LATENCY_REPEATS` times per query, after a discarded warm-up
> pass — the first call loads weights and allocates buffers, and including it would swamp the
> numbers. Hardware and device are printed with the table because these are wall-clock figures
> from one laptop, not published benchmarks.

## Cell 26 — Per-stage timing

**What** Defines `time_ms`, times five stages across all queries, appends the LLM row, builds
the `latency` DataFrame with the `hardware` column, writes `results/latency.csv`, and prints the
`p1` / `p2` comparison used by cell 28.

**Why** Stages are timed separately so the table shows *where* the cost is, not just a total.

**PDF** §4.3 (record and tabulate baseline vs re-ranking runtime).

**Gotchas**
- `time_ms` discards a warm-up call; the first call loads weights and would dominate.
- LLM timings are read from the answer cache, where every entry stored the duration of the real
  request that produced it. That covers all generations rather than only the few that missed the
  cache on a given run — an earlier version read `t > 0` from the current run and dropped the row
  entirely once every answer was cached.
- The row still spans more than one model and has high variance; the `note` column in
  `latency.csv` carries that caveat alongside the data.

## Cell 27 — Scaling heading

**What** Markdown only, short.

**PDF** §4.4.

## Cell 28 — Cross-encoder throughput and extrapolation

**What** Scores the entire corpus against one query to measure `pairs_per_sec`, extrapolates to
172 / 10,000 / 1,000,000 chunks, writes `results/scaling.csv`, and prints the mechanism.

**Why** The §4.4 answer is measured on this machine rather than asserted from theory: `BULK` is
the whole corpus, which is literally the thing the section argues against doing per query.

**PDF** §4.4 (why not cross-encode the entire collection).

**Gotcha** The bi-encoder embeds each chunk once at index time; the cross-encoder cannot
precompute anything because its score depends on query and chunk jointly. That is why its cost
is O(N) per query and why the two-stage design confines the O(N) stage to N=10.

## Cell 29 — Outputs index

**What** Markdown only. Table of everything written to `results/`.

**PDF** Packaging for §4.

> *Preserved (original markdown):*
>
> ## 11. Outputs
>
> | file | contents |
> |---|---|
> | `results/chunks.csv` | the corpus: chunk id, section, rule numbers, pages, token count, text |
> | `results/retrieval_runs.csv` | every candidate of every query with both ranks and both scores |
> | `results/query_summary.csv` | one row per query: gold ranks, outcome, Top 3 sets |
> | `results/rank_shifts.csv` | gold-chunk rank before and after re-ranking |
> | `results/near_duplicates.csv` | overlap-induced and document-level duplicate candidates |
> | `results/answers.md` | Answer 1 vs Answer 2, side by side |
> | `results/answers_cache.json` | frozen generations keyed by (model, prompt hash) |
> | `results/latency.csv` | per-stage timings, mean ± std, with hardware |
> | `results/scaling.csv` | cross-encoder throughput and the extrapolation |
>
> `NOTES.md` records why each decision was made and what was observed, including the cases where
> re-ranking changed nothing. The report is written by hand from these files.
