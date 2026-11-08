"""Task 2.3: call the local LLM (Qwen3-4B-Instruct-2507) through Ollama's native /api/chat.

Every option is passed explicitly in each call (never rely on Ollama/Modelfile defaults).
Uses only the Python standard library (urllib), no new dependency.
"""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# Main setting: temperature 0 + fixed seed. Values are starting values.
MAIN_OPTIONS = {
    "temperature": 0.0,
    "seed": 42,
    "num_ctx": 4096,       # placeholder for the Step 3 test; real value chosen from data in Step 5
    "num_predict": 256,    # short answers (starting value)
}
# Secondary setting: Qwen's recommended values (seed is set per run).
QWEN_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "num_ctx": 4096,
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


if __name__ == "__main__":
    msgs = [{"role": "user",
             "content": "Context: Paris is the capital of France.\n"
                        "Question: What is the capital of France?\n"
                        "Answer with a short answer only."}]
    resp = chat(msgs, MAIN_OPTIONS)
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    content = resp["message"]["content"]
    print("\n<think> in answer:", "<think>" in content)
