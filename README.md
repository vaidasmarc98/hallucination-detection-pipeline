# Hallucination Detection Pipeline

An evaluation pipeline that measures how often an LLM fabricates information, using real-world trivia questions as ground truth. Built with Python, DeepEval, and the HuggingFace `datasets` library.

---

## What Problem This Solves

Hallucination is one of the most critical failure modes in production LLM systems. This pipeline provides a repeatable, data-driven way to measure hallucination rate and faithfulness across a large sample of questions — generating structured reports that can be tracked over time or across model versions.

---

## Features

- **Real dataset integration** — pulls questions from TriviaQA via HuggingFace
- **Hallucination scoring** — DeepEval `HallucinationMetric` with configurable thresholds
- **Faithfulness scoring** — DeepEval `FaithfulnessMetric` against ground-truth context
- **Structured JSON reports** — timestamped, includes per-question scores and reasons
- **Summary statistics** — pass rates, averages, failure counts at a glance

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| [DeepEval](https://github.com/confident-ai/deepeval) | Hallucination + faithfulness metrics |
| Anthropic API | Model under evaluation (Claude Haiku) |
| HuggingFace `datasets` | TriviaQA ground-truth data |
| pandas | Optional report analysis |

---

## Project Structure

```
hallucination-detector/
├── data/
│   └── load_dataset.py          # TriviaQA loader
├── evaluator/
│   └── hallucination_eval.py    # core pipeline
├── reports/                     # JSON outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set API keys**
```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here   # used by DeepEval's LLM-as-judge
```

**3. Run the pipeline**
```bash
python evaluator/hallucination_eval.py
```

By default evaluates 20 samples. To change the sample size:
```python
run_pipeline(num_samples=50)
```

---

## Example Output

```
Loading 20 TriviaQA samples...
  [1/20] Which American-born Physicist won the Nobel P...
    ✓ hallucination=0.12  faithfulness=0.91
  [2/20] In which sport would you find a 'Crosse'?
    ✓ hallucination=0.08  faithfulness=0.95
  [3/20] Who wrote the 1954 novel 'Lord of the Flies'?
    ✗ hallucination=0.61  faithfulness=0.44

── Summary ───────────────────────────────────────────
  Total evaluated:         20
  Hallucination pass rate: 85.0%
  Faithfulness pass rate:  80.0%
  Avg hallucination score: 0.183
  Avg faithfulness score:  0.847

  Report saved → reports/results_20260316_091423.json
```

---

## Report Schema

Each `results_*.json` file contains:
```json
{
  "timestamp": "20260316_091423",
  "model": "claude-haiku-4-5-20251001",
  "summary": { ... },
  "results": [
    {
      "question": "...",
      "ground_truth": "...",
      "response": "...",
      "hallucination_score": 0.12,
      "hallucination_passed": true,
      "faithfulness_score": 0.91,
      "faithfulness_passed": true,
      "hallucination_reason": "...",
      "faithfulness_reason": "..."
    }
  ]
}
```

---

## Author

Vaidas Marcinkevicius · [linkedin.com/in/vaidasmarc](https://linkedin.com/in/vaidasmarc)
