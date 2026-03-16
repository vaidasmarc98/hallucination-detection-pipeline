"""
Hallucination Detection Pipeline
---------------------------------
For each question:
  1. Query Claude for an answer
  2. Score the response using DeepEval's HallucinationMetric and FaithfulnessMetric
  3. Log results to reports/results.json
"""

import json
import os
import sys
from datetime import datetime

from anthropic import Anthropic
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.load_dataset import load_trivia_qa

# ── setup ─────────────────────────────────────────────────────────────────────

client = Anthropic()
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── LLM call ──────────────────────────────────────────────────────────────────

def query_claude(question: str) -> str:
    """Send a question to Claude and return the raw text response."""
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Answer the following question concisely and factually. "
                    f"If you are not certain, say so.\n\nQuestion: {question}"
                ),
            }
        ],
    )
    return message.content[0].text.strip()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_response(question: str, ground_truth: str, response: str) -> dict:
    """
    Run HallucinationMetric and FaithfulnessMetric against a single response.
    Returns a result dict with scores and pass/fail status.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=ground_truth,
        context=[ground_truth],          # ground truth as the reference context
        retrieval_context=[ground_truth],
    )

    hallucination_metric = HallucinationMetric(threshold=0.5)
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    hallucination_metric.measure(test_case)
    faithfulness_metric.measure(test_case)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "response": response,
        "hallucination_score": hallucination_metric.score,
        "hallucination_passed": hallucination_metric.is_successful(),
        "faithfulness_score": faithfulness_metric.score,
        "faithfulness_passed": faithfulness_metric.is_successful(),
        "hallucination_reason": hallucination_metric.reason,
        "faithfulness_reason": faithfulness_metric.reason,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def build_summary(results: list[dict]) -> dict:
    total = len(results)
    hallucination_passes = sum(1 for r in results if r["hallucination_passed"])
    faithfulness_passes = sum(1 for r in results if r["faithfulness_passed"])
    avg_hallucination = sum(r["hallucination_score"] for r in results) / total
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / total

    return {
        "total_evaluated": total,
        "hallucination_pass_rate": round(hallucination_passes / total, 3),
        "faithfulness_pass_rate": round(faithfulness_passes / total, 3),
        "avg_hallucination_score": round(avg_hallucination, 3),
        "avg_faithfulness_score": round(avg_faithfulness, 3),
        "hallucination_failures": total - hallucination_passes,
        "faithfulness_failures": total - faithfulness_passes,
    }


def save_report(results: list[dict], summary: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"results_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "model": "claude-haiku-4-5-20251001",
        "summary": summary,
        "results": results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


# ── main ──────────────────────────────────────────────────────────────────────

def run_pipeline(num_samples: int = 20):
    print(f"Loading {num_samples} TriviaQA samples...")
    samples = load_trivia_qa(num_samples)

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{num_samples}] {sample['question'][:60]}...")
        response = query_claude(sample["question"])
        result = evaluate_response(sample["question"], sample["answer"], response)
        results.append(result)

        status = "✓" if result["hallucination_passed"] and result["faithfulness_passed"] else "✗"
        print(f"    {status} hallucination={result['hallucination_score']:.2f}  "
              f"faithfulness={result['faithfulness_score']:.2f}")

    summary = build_summary(results)
    report_path = save_report(results, summary)

    print("\n── Summary ───────────────────────────────────────────")
    print(f"  Total evaluated:        {summary['total_evaluated']}")
    print(f"  Hallucination pass rate: {summary['hallucination_pass_rate']*100:.1f}%")
    print(f"  Faithfulness pass rate:  {summary['faithfulness_pass_rate']*100:.1f}%")
    print(f"  Avg hallucination score: {summary['avg_hallucination_score']}")
    print(f"  Avg faithfulness score:  {summary['avg_faithfulness_score']}")
    print(f"\n  Report saved → {report_path}")

    return results, summary


if __name__ == "__main__":
    run_pipeline(num_samples=20)
