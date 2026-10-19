# PLAN.md — RAG Assignment: Baseline vs. Re-Ranking

**Source of truth:** `RAG_LMA_Assignment_v1.pdf`. **Deadline: today, 19 Oct 2026.**
**Deliverables:** (1) GitHub link to a repo whose primary artifact is a single well-commented
notebook, (2) a concise report, **written by the human, not AI-generated**.

**Time-box:** a few hours to a finished, restart-and-run-all notebook, then stop. Scope is the
PDF and nothing more. Revision 2 (amendments of 19 Oct 2026 folded in).

---

## 1. Objectives (mapping to the PDF)

| # | PDF section | Objective | Done when |
|---|---|---|---|
| O1 | §1 Setup | One LLM for generation, one bi-encoder, one complex PDF corpus | Env reproducible from `requirements.txt`; download cell fetches the corpus |
| O2 | §2 Pipeline 1 | Index chunks with a bi-encoder; retrieve **Top 10** by cosine/inner product; feed **Top 3** to the LLM → **Answer 1** | top-10 with scores; Answer 1 for every query |
| O3 | §2 Pipeline 2 | **Same Top 10** → cross-encoder scores all 10 `(query, chunk)` pairs → re-sort desc → **Top 3** → **identical prompt** → **Answer 2** | Both answers differ only in retrieved context |
| O4 | §3 Failure modes | **≥2** clean promotion cases + one per named category **where one genuinely exists**; no-ops and regressions reported honestly | 6–8 hand-labelled queries with evidence tables |
| O5 | §4.1 Architecture | Bi-encoder (independent embeddings, cosine) vs cross-encoder (joint self-attention) | Notes + evidence in `NOTES.md` |
| O6 | §4.2 Rank shift | Rank changes ("Chunk X: #8 → #1"); Answer 1 vs 2 compared for hallucination / omitted clause / misstated rule | `results/rank_shifts.csv` + `results/answers.md` |
| O7 | §4.3 Latency | Tabulated runtime, baseline vs re-ranking | `results/latency.csv`, mean ± std, hardware + device stated |
| O8 | §4.4 Scaling | Why not cross-encode the whole collection | Answered with **measured** CE pairs/sec extrapolation |

**Non-goals** (not in the PDF, so not built): hybrid BM25, query rewriting, chunk-size
ablations, a UI, RAGAS, fine-tuning, any extra pipeline variant.

---

## 2. Locked decisions

| Decision | Choice | Why (full reasoning in `NOTES.md`) |
|---|---|---|
| Generation LLM | **Gemini**, model id in **one config variable** `GEN_MODEL_CANDIDATES = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]`; smoke test picks the first that responds | `gemini-2.5-flash` passed its 16 Oct 2026 shutdown. One variable = one place to change |
| Temperature | **Left at default** (not set) | Google's Gemini 3 docs strongly recommend not changing `temperature`/`top_p`/`top_k`; reasoning is tuned for defaults. The "only context differs" claim rests instead on a **byte-identical prompt template** + a **persistent answer cache** |
| Thinking | **`thinking_level="minimal"`** if the installed SDK accepts it | Lower latency and cleaner §4.3 timings; minimal ≈ "no thinking" |
| LLM call budget | **≤18 total**, every answer cached to `results/answers_cache.json` keyed by `(model, sha256(prompt))`; cached answers are **never** regenerated | Free tier may be ~20 requests/day. Re-running the notebook must cost 0 calls |
| Corpus | **Twilight Imperium 4th Ed. "Living Rules Reference" v1.3** (official FFG PDF) | Densest available supply of "except when", role limits and prohibitions — the §3 failure modes. Source URL in `NOTES.md`; `data/*.pdf` gitignored; notebook has a download cell |
| Chunking | Split on **numbered rule boundaries** where the layout allows; **tokenizer-capped under 256 word-pieces**; store **rule number + page** per chunk | §2 needs one shared corpus; rule numbers make every retrieved chunk citable in the report |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, fast. **Comparable size** to the mandated cross-encoder (both 6-layer MiniLM), so the comparison is not confounded by a 10× parameter gap — but it is **not** a clean architecture-only comparison: the two models differ in training data and objective (general sentence similarity vs MS MARCO relevance ranking) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Named in the PDF |
| Vector store | FAISS `IndexFlatIP` over L2-normalised vectors | IP on normalised vectors **is** cosine, satisfying §2.b either way. **Exact** search, so a chunk missing from the top-10 is bi-encoder semantics, never ANN approximation error |
| Runtime | Python **3.12** venv via `uv` | Chosen for **stability** — 3.12 is the best-supported version for the torch / sentence-transformers / faiss stack; avoids spending deadline time on bleeding-edge wheel problems |
| Tabular output | **CSV** | Readable in the repo diff and in any spreadsheet; no parquet engine dependency |
| Code shape | All logic in one notebook, no `src/` package | PDF strongly prefers a single well-commented notebook |
| Documentation | **One `NOTES.md`** (decisions, observations, surprises — evidence only, no report prose) | Replaces the earlier `Advisor/` folder. The human writes the report from it |

---

## 3. Repository layout

```
LMA_RAG/
  RAG_LMA_Assignment_v1.pdf                   # the brief
  PLAN.md                                     # this file
  NOTES.md                                    # decisions + observations (report input)
  README.md                                   # setup + how to run
  requirements.txt
  .env.example                                # GEMINI_API_KEY=
  .gitignore                                  # .venv, .env, data/*.pdf, caches
  data/                                       # corpus PDF (gitignored, download cell fetches it)
  notebooks/rag_baseline_vs_reranking.ipynb   # PRIMARY DELIVERABLE
  results/
    chunks.csv                                # chunk_id, rule_no, page, n_tokens, text
    retrieval_runs.csv                        # query_id, chunk_id, bi_rank, bi_score, ce_rank, ce_score, is_gold
    rank_shifts.csv                           # per query: gold chunk #before -> #after
    answers.md                                # Answer 1 vs Answer 2, side by side
    answers_cache.json                        # (model, prompt hash) -> answer
    latency.csv                               # per-stage timings, mean ± std
    near_duplicates.csv                       # overlap-induced near-dupes seen in top-10
```

---

## 4. Execution plan

### Phase 0 — Environment
- [x] `uv venv --python 3.12 .venv`; install torch, sentence-transformers, faiss-cpu, pymupdf, google-genai, python-dotenv, pandas, jupyter, ipykernel, nbformat; freeze `requirements.txt`.
- [x] `.env.example`, `.gitignore`. Key via `python-dotenv`; never printed (the notebook is committed **with** outputs).
- [x] Smoke test: load both local models, report device; resolve `GEN_MODEL` from the candidate list with 1 cheap call; confirm whether the installed SDK accepts `thinking_level`.

### Phase 1 — Corpus & chunking
- [x] Download cell → `data/ti4_living_rules_reference_v1_3_web.pdf` (URL recorded in `NOTES.md`).
- [x] PyMuPDF page-aware extraction; log what the 2-column layout and iconography do to the text.
- [x] Chunk on numbered rule boundaries (`"12.3 ..."` style); carry `rule_no` + `page`; **hard-verify every chunk is < 256 word-pieces** with the bi-encoder's own tokenizer — above that the model silently truncates and text invisible to retrieval would still be visible to the LLM.
- [x] Persist `results/chunks.csv`.

### Phase 2 — Pipeline 1 (baseline)
- [x] Embed, L2-normalise, FAISS `IndexFlatIP`.
- [x] `baseline_retrieve(query) -> top 10 + scores`; this exact list is handed to Pipeline 2.
- [x] One fixed prompt template: context-only instruction + 3 chunks + question. "Answer only from the context; say explicitly if it is insufficient" — so an omission surfaces as an omission instead of being masked by model prior knowledge.
- [x] Answer 1 from top-3 (cache-first).

### Phase 3 — Pipeline 2 (re-ranking)
- [x] Cross-encoder scores the **same 10** pairs; re-sort desc; top-3.
- [x] Answer 2 via the **byte-identical** template; assert the candidate sets match.
- [x] Note: when re-ranking leaves the top-3 set unchanged, the prompt hash is identical, so the cache returns Answer 1 — zero extra calls and a provably identical answer.

### Phase 4 — Query set & failure hunting
- [x] **6–8 hand-labelled queries** with known gold rule numbers, aimed at the four named categories: keyword trap/distractor, exceptions & corner cases, negation & constraints, vital chunk buried at 4–10.
- [x] Per query record: gold in baseline top-3? gold in rerank top-3? → promotion / no-op / regression.
- [x] Keep **≥2 clean promotions**; **report no-ops and regressions as findings**, not failures to hide.
- [x] **Log near-duplicates:** overlapping chunks can put two near-identical texts in the top-10, wasting candidate slots and context budget. Record every occurrence to `results/near_duplicates.csv` (high cosine between distinct retrieved chunk texts).

### Phase 5 — Analysis artifacts
- [x] `rank_shifts.csv` + readable table: `Rule 61.3 (p.44) #8 → #1`.
- [x] `answers.md`: side-by-side Answer 1 / Answer 2 with the defect named per case (hallucinated rule, omitted clause, misstated restriction).

### Phase 6 — Latency profiling
- [x] Per-stage timing: query embed, FAISS search, CE scoring of 10 pairs, LLM call. `n=10` repeats after a discarded warm-up (first call loads weights and would dominate).
- [x] Report retrieval-only **and** end-to-end; state hardware and device (CPU vs MPS). Cached LLM answers are excluded from LLM timing and that is stated.

### Phase 7 — Scaling answer
- [x] Measure CE pairs/sec on this machine; extrapolate to the full corpus and to 1M chunks; contrast O(N) joint encoding per query with one-time embedding + sublinear ANN.

### Phase 8 — Packaging
- [x] Execute the notebook top-to-bottom from a clean kernel (`nbconvert --execute`) so restart-and-run-all is proven, not assumed.
- [x] `README.md`, frozen `requirements.txt`, repo committed and link-ready.
- [x] Hand over `NOTES.md` + `results/`; the human writes the report.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Model id dead or renamed | Candidate list + smoke test; one config variable to change |
| Free-tier quota (~20/day) | ≤18 calls budgeted; persistent answer cache; identical prompts dedupe automatically |
| Chunks exceed 256 word-pieces → silent truncation | Tokenizer-verified cap in Phase 1, asserted not assumed |
| Rule-boundary regex fails on 2-column layout | Fall back to token-window chunking for sections that do not parse; log how many chunks came from which path |
| A named failure category yields nothing | Say so explicitly; do not invent a result |
| Re-ranking hurts on some query | Report it — the PDF asks for analysis, not a win rate |
| Rulebook redistribution | `data/*.pdf` gitignored; only the URL + download cell are committed |
| Deadline | Phases 0–3 are the critical path; 4–7 are additive. Stop at a finished notebook |


---

## 6. Outcome and deviations from this plan

All eight phases completed; the notebook runs top-to-bottom from a clean kernel
(`nbconvert --execute`, exit 0) and every table in `results/` is generated by it.

Four deviations, all recorded in `NOTES.md` with reasoning:

1. **Query count 8 → 10.** Eight hand-written queries produced only one clean promotion, below
   the assignment's floor of two. §4b of the notebook screens 16 further candidates through
   retrieval only (zero API cost) and found five; Q9 and Q10 were adopted from it. Final tally:
   **3 promotions, 1 rank regression, 1 no-op, 5 context-changed-verdict-same**.
2. **`LLM_CALL_BUDGET` 18 → 22**, to cover the two added queries. The final clean run used only
   **4** calls (17 cache hits). Across the whole session, including the aborted runs that
   debugged quota handling, roughly **30** requests were actually spent — see NOTES.md 3.6,
   which states this plainly rather than reporting only the per-run figure.
3. **Model fail-over.** `gemini-3.5-flash` exhausted its allowance mid-project, so Q9 and Q10
   were answered by `gemini-3.1-flash-lite`. Each query's Answer 1 / Answer 2 pair is always
   from a single model (asserted); the model is recorded per query.
4. **Pinned `torch==2.5.1` / `pymupdf==1.27.2`** to halve download size under the deadline,
   which then forced `transformers<5` for compatibility. Reproducibility note in NOTES.md 3.5.
