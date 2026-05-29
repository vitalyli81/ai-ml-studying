"""Eval harness — grade the RAG chatbot against a golden dataset.

Two scoring layers:
  1. Deterministic checks (keywords, correct refusal)
  2. LLM-as-judge groundedness (is the answer supported by context?)

See README.md for the upgrades. Concepts: ../../llms/evals.md
"""

import json
import sys
from pathlib import Path

from anthropic import Anthropic

# The pipeline under test. Point this at your own ask() to eval anything.
sys.path.insert(0, str(Path(__file__).parent.parent / "rag-chatbot"))
import rag  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
JUDGE_MODEL = "claude-opus-4-7"  # use a stronger model to judge a weaker one

client = Anthropic()


def load_golden() -> list[dict]:
    with GOLDEN_SET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check_deterministic(example: dict, answer: str) -> tuple[bool, str]:
    """Fast, free, exact checks. Run these before spending money on a judge."""
    low = answer.lower()
    for needle in example.get("expected_contains", []):
        if needle.lower() not in low:
            return False, f"missing expected substring: {needle!r}"
    return True, "ok"


def judge_groundedness(question: str, answer: str, context: str) -> tuple[bool, str]:
    """LLM-as-judge: is every claim in the answer supported by the context?
    Catches hallucination that keyword checks miss."""
    prompt = (
        "Given the CONTEXT and ANSWER, is every factual claim in the ANSWER "
        "supported by the CONTEXT? An honest 'I don't know' counts as grounded.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        'Reply with JSON only: {"grounded": true|false, "reason": "..."}'
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        verdict = json.loads(resp.content[0].text)
        return bool(verdict["grounded"]), verdict.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        return False, "judge returned unparseable output"


def run() -> int:
    rag.index_docs()
    examples = load_golden()
    passed = 0

    print(f"Running {len(examples)} examples\n" + "=" * 60)
    for ex in examples:
        # Retrieve context here too, so the judge sees what the model saw.
        retrieved = rag.collection.query(query_texts=[ex["input"]], n_results=rag.TOP_K)
        context = "\n\n".join(retrieved["documents"][0])

        result = rag.ask(ex["input"])
        answer = result["answer"]

        det_ok, det_msg = check_deterministic(ex, answer)
        grounded, judge_msg = judge_groundedness(ex["input"], answer, context)
        ok = det_ok and grounded
        passed += ok

        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {ex['id']:24} tags={ex.get('tags', [])}")
        if not det_ok:
            print(f"       deterministic: {det_msg}")
        if not grounded:
            print(f"       groundedness:  {judge_msg}")

    rate = passed / len(examples)
    print("=" * 60)
    print(f"Pass rate: {passed}/{len(examples)} = {rate:.0%}")

    # The eval gate: exit non-zero so CI can block a regression.
    THRESHOLD = 0.80
    if rate < THRESHOLD:
        print(f"BELOW THRESHOLD ({THRESHOLD:.0%}) — would block deploy.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
