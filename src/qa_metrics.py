"""Answer metrics: exact match (EM) and F1, following the official HotpotQA evaluation.

normalize: lowercase, remove punctuation, remove articles (a, an, the), collapse spaces.
F1 special case (as in HotpotQA): if either side is yes / no / noanswer and they differ, F1 = 0.
"""
import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (p in special or g in special) and p != g:
        return 0.0
    p_tok, g_tok = p.split(), g.split()
    common = Counter(p_tok) & Counter(g_tok)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p_tok), same / len(g_tok)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert abs(f1_score("Greenwich Village", "Greenwich Village, New York City") - 0.5714) < 1e-3
    print("qa_metrics: all checks pass")
