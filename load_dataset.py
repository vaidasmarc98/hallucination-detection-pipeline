"""
Load a sample of TriviaQA questions from HuggingFace datasets.
Returns a list of dicts with 'question' and 'answer' keys.
"""

from datasets import load_dataset


def load_trivia_qa(num_samples: int = 50) -> list[dict]:
    """
    Pull num_samples questions from the TriviaQA validation split.
    Each record: { "question": str, "answer": str }
    """
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    samples = []

    for row in dataset.select(range(num_samples)):
        # TriviaQA stores multiple valid answers — take the first normalized alias
        answer = row["answer"]["normalized_aliases"][0]
        samples.append({
            "question": row["question"],
            "answer": answer,
        })

    return samples


if __name__ == "__main__":
    samples = load_trivia_qa(5)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}\n")
