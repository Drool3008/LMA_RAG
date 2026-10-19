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

