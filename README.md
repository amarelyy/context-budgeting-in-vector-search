# Index-Level Context Budgeting in Vector Search

**Goal:** minimize the context budget (in tokens) that a vector database feeds to an LLM in a RAG system, without losing the passage that holds the answer.

Instead of a fixed number of passages (k), retrieval picks the most relevant chunks that fit inside a token budget. The budget is calibrated from data with conformal prediction.

> Status: work in progress. No results yet.

## Results

| Method | recall_at_budget | avg total_tokens | tokens_saved_pct |
|---|---|---|---|
| Fixed-k (k=5) | not run yet | not run yet | n/a |
| Fixed-k (k=10) | not run yet | not run yet | n/a |
| Fixed-k (k=20) | not run yet | not run yet | n/a |
| Token-budget (conformal) | not run yet | not run yet | not run yet |

Every number will come from `results/results.md`.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
