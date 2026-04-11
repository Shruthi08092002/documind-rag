import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datasets import Dataset

from tests.test_set import TEST_SET
from src.retriever import retrieve
from src.generator import generate_answer

RESULTS_PATH = "tests/eval_results.json"


def score_faithfulness(answer: str, context: list) -> float:
    """
    Measures if the answer only uses information from context.
    Simple heuristic: what fraction of answer sentences
    contain words that appear in the context?
    """
    context_text = " ".join(context).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        context_words = set(context_text.split())
        overlap = words & context_words
        if len(overlap) / max(len(words), 1) > 0.3:
            supported += 1
    return round(supported / len(sentences), 3)


def score_answer_relevancy(answer: str, question: str) -> float:
    """
    Measures if the answer addresses the question.
    Heuristic: keyword overlap between question and answer.
    """
    stopwords = {"what", "is", "the", "a", "an", "of", "in",
                 "are", "how", "does", "do", "why", "which"}
    question_words = set(question.lower().split()) - stopwords
    answer_words = set(answer.lower().split())
    if not question_words:
        return 0.0
    overlap = question_words & answer_words
    return round(len(overlap) / len(question_words), 3)


def score_context_recall(context: list, ground_truth: str) -> float:
    """
    Measures if retrieved context contains ground truth info.
    Heuristic: keyword overlap between ground truth and context.
    """
    context_text = " ".join(context).lower()
    gt_words = set(ground_truth.lower().split())
    stopwords = {"what", "is", "the", "a", "an", "of", "in",
                 "are", "how", "does", "do", "why", "which",
                 "that", "this", "it", "by", "for", "with"}
    gt_words = gt_words - stopwords
    if not gt_words:
        return 0.0
    context_words = set(context_text.split())
    overlap = gt_words & context_words
    return round(len(overlap) / len(gt_words), 3)


def run_evaluation(test_set: list, config_name: str = "baseline") -> dict:
    print(f"\n{'='*60}")
    print(f"Running evaluation: {config_name}")
    print(f"Questions: {len(test_set)}")
    print(f"{'='*60}\n")

    faithfulness_scores = []
    relevancy_scores = []
    recall_scores = []

    for i, item in enumerate(test_set):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(test_set)}] {question[:55]}...")

        try:
            result = generate_answer(question)
            answer = result["answer"]

            docs = retrieve(question, k=3)
            context = [doc.page_content for doc in docs]

            f = score_faithfulness(answer, context)
            r = score_answer_relevancy(answer, question)
            c = score_context_recall(context, ground_truth)

            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            recall_scores.append(c)

            print(f"         faith={f:.2f} relevancy={r:.2f} recall={c:.2f}")

        except Exception as e:
            print(f"         Error: {e}")
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)
            recall_scores.append(0.0)

    avg_faith = round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
    avg_rel = round(sum(relevancy_scores) / len(relevancy_scores), 3)
    avg_rec = round(sum(recall_scores) / len(recall_scores), 3)

    results = {
        "config": config_name,
        "num_questions": len(test_set),
        "faithfulness": avg_faith,
        "answer_relevancy": avg_rel,
        "context_recall": avg_rec,
    }

    existing = []
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)

    existing.append(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS — {config_name}")
    print(f"{'='*60}")
    print(f"  Faithfulness:      {avg_faith}")
    print(f"  Answer relevancy:  {avg_rel}")
    print(f"  Context recall:    {avg_rec}")
    print(f"\nSaved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    results = run_evaluation(TEST_SET, config_name="baseline_k3")