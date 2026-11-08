"""Task 2.3: call the local LLM (Qwen3-4B-Instruct-2507) through Ollama's native /api/chat.

Every option is passed explicitly in each call (never rely on Ollama/Modelfile defaults).
Uses only the Python standard library (urllib), no new dependency.
"""
import json
import urllib.request

# 127.0.0.1, not "localhost": on Windows, localhost tries IPv6 first and adds ~2.2 s per call
# (measured 2026-11-08: 2.4 s vs 0.15 s for the same request).
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# Main setting: temperature 0 + fixed seed. Values are starting values.
MAIN_OPTIONS = {
    "temperature": 0.0,
    "seed": 42,
    "num_ctx": 5120,       # Step 5: max k=20 prompt 4599 + num_predict 256 = 4855, rounded up to 1024
    "num_predict": 256,    # short answers (starting value)
}
# Secondary setting: Qwen's recommended values (seed is set per run).
QWEN_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "num_ctx": 5120,       # SAME num_ctx as the main setting
    "num_predict": 256,
}


def chat(messages, options, model=MODEL, timeout=600):
    """Send one chat request. Returns the full JSON response (dict)."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


_TOKENIZER = None


def count_prompt_tokens(messages):
    """Our own count: Qwen3 tokenizer + chat template, the same way Ollama builds the prompt.
    Checked in Step 4: equals Ollama's prompt_eval_count exactly (485 = 485)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        from chunking import TOKENIZER_NAME
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    out = _TOKENIZER.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=True, return_dict=True)
    return len(out["input_ids"])


def chat_checked(messages, options):
    """chat() + truncation guard. Returns (response, info dict).

    Step 4 findings (Ollama 0.40.15): a prompt longer than num_ctx gives HTTP 400
    (no silent truncation), and prompt caching does NOT lower prompt_eval_count.
    The guard still flags any case where Ollama read fewer tokens than we sent.
    """
    ours = count_prompt_tokens(messages)
    if ours + options["num_predict"] > options["num_ctx"]:
        raise ValueError(f"prompt ({ours}) + num_predict ({options['num_predict']}) "
                         f"> num_ctx ({options['num_ctx']})")
    resp = chat(messages, options)
    seen = resp["prompt_eval_count"]
    info = {"prompt_tokens_ours": ours, "prompt_eval_count": seen,
            "truncated": seen < ours, "done_reason": resp["done_reason"]}
    return resp, info


if __name__ == "__main__":
    msgs = [{"role": "user",
             "content": "Context: Paris is the capital of France.\n"
                        "Question: What is the capital of France?\n"
                        "Answer with a short answer only."}]
    resp = chat(msgs, MAIN_OPTIONS)
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    content = resp["message"]["content"]
    print("\n<think> in answer:", "<think>" in content)
