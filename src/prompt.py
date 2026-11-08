"""Task 2.3: the ONE prompt template used for every method (fixed-k, k=0, token-budget).

Passages are fed as their text only (no extra title line), so the tokens of each
passage equal its token_cost. Extra tokens come only from this template and the
chat template (measured as "overhead" in the truncation guard).
Do not change this template between runs; if it changes, rerun everything.
"""

INSTRUCTION = (
    "Answer the question using the passages below if they help. "
    "Give a short answer only (a few words, a name, a number, or yes/no). "
    "Do not explain."
)
INSTRUCTION_NO_CONTEXT = (
    "Answer the question. "
    "Give a short answer only (a few words, a name, a number, or yes/no). "
    "Do not explain."
)


def build_messages(question, passages):
    """passages: list of chunk texts in ranked order (empty list = no retrieval, k=0)."""
    if passages:
        ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        content = f"{INSTRUCTION}\n\nPassages:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    else:
        content = f"{INSTRUCTION_NO_CONTEXT}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "user", "content": content}]
