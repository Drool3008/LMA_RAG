# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines compared over one chunked corpus — the **Twilight Imperium 4th Edition
Living Rules Reference** (v1.3, official Fantasy Flight Games PDF, 38 pages).

| | Pipeline 1 — baseline | Pipeline 2 — re-ranking |
|---|---|---|
| Candidate generation | bi-encoder + FAISS, **Top 10** by cosine | **the same Top 10** |
| Selection | **Top 3** as retrieved | all 10 `(query, chunk)` pairs scored by a cross-encoder, re-sorted, **Top 3** |
| Generation | Gemini, fixed prompt template → **Answer 1** | **identical** template → **Answer 2** |

Models: `sentence-transformers/all-MiniLM-L6-v2` (bi-encoder),
`cross-encoder/ms-marco-MiniLM-L-6-v2` (cross-encoder), Gemini Flash (generation).

## Repository

| path | what it is |
|---|---|
| `notebooks/rag_baseline_vs_reranking.ipynb` | **the deliverable** — both pipelines, failure-mode analysis, latency, scaling |
| `docs/CELL_GUIDE.md` | cell-by-cell walkthrough of the notebook: what each cell does, why, which assignment section it serves, and the gotchas |
| `results/` | generated CSVs + `answers.md` (Answer 1 vs Answer 2) |
| `RAG_LMA_Assignment_v1.pdf` | the assignment brief |

The working notes the report is written from (`NOTES.md`, `PLAN.md`) are deliberately kept
out of the repo — the report has to be in the author's own words, so the decision log stays
local. `docs/CELL_GUIDE.md` covers everything needed to read the code itself.

## Setup

Python **3.12** (the version this stack is best supported on):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # then paste your key: GEMINI_API_KEY=...
.venv/bin/python -m ipykernel install --user --name lma_rag --display-name "Python 3 (LMA_RAG .venv)"
```

Without `uv`, plain venv and pip do the same job:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your key: GEMINI_API_KEY=...
.venv/bin/python -m ipykernel install --user --name lma_rag --display-name "Python 3 (LMA_RAG .venv)"
```

Then open `notebooks/rag_baseline_vs_reranking.ipynb` in VS Code or JupyterLab and select the
**Python 3 (LMA_RAG .venv)** kernel. For a local JupyterLab server, add it separately — it is
not in `requirements.txt` because the notebook does not need it to run:

```bash
uv pip install --python .venv/bin/python jupyterlab
.venv/bin/jupyter lab notebooks/rag_baseline_vs_reranking.ipynb
```

To re-execute headlessly (this is how the committed outputs were produced):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/rag_baseline_vs_reranking.ipynb
```

Get a free Gemini key at <https://aistudio.google.com/apikey>.

The notebook downloads the corpus PDF on first run (`data/*.pdf` is gitignored, so the
publisher's file is not redistributed here — only the URL and the loader).

## Notes on running it

- **Restart-and-run-all is the supported path**; the committed outputs came from exactly that.
- **API quota:** every generated answer is cached to `results/answers_cache.json` keyed by
  `(model, sha256(prompt))` and never regenerated, so re-running costs **zero** API calls.
  If no model responds at all (quota spent, or the service is briefly down), the notebook says so
  and still runs end to end on the cached answers.
  A fresh run with an empty cache uses at most 18.
- **Model ids expire.** The generation model lives in one variable,
  `GEN_MODEL_CANDIDATES`; the smoke test picks the first candidate that responds.
- First run downloads ~120 MB of model weights from Hugging Face.
