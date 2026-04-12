import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from tests.test_set import TEST_SET
from src.generator import generate_answer
from src.retriever import retrieve
from tests.evaluator import (
    run_evaluation,
    score_faithfulness,
    score_answer_relevancy,
    score_context_recall,
    RESULTS_PATH
)


def run_experiment(test_set: list, config_name: str,
                   k: int = 3, search_type: str = "similarity") -> dict:
    """
    Runs evaluation with specific retrieval settings.
    Same as run_evaluation but lets us control k and search_type.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {config_name}")
    print(f"Settings: k={k}, search_type={search_type}")
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

            docs = retrieve(question, k=k, search_type=search_type)
            context = [doc.page_content for doc in docs]

            f = score_faithfulness(answer, context)
            r = score_answer_relevancy(answer, question)
            c = score_context_recall(context, ground_truth)

            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            recall_scores.append(c)

            print(f"         faith={f:.2f} rel={r:.2f} recall={c:.2f}")

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
        "k": k,
        "search_type": search_type,
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

    print(f"\n  Faithfulness:     {avg_faith}")
    print(f"  Answer relevancy: {avg_rel}")
    print(f"  Context recall:   {avg_rec}")

    return results


def print_comparison():
    """Prints a comparison table of all experiment results."""
    if not Path(RESULTS_PATH).exists():
        print("No results found yet.")
        return

    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Faithful':>10} {'Relevancy':>10} {'Recall':>10}")
    print("-" * 55)

    for r in all_results:
        print(f"{r['config']:<20} "
              f"{r['faithfulness']:>10} "
              f"{r['answer_relevancy']:>10} "
              f"{r['context_recall']:>10}")

    print(f"{'='*60}")


if __name__ == "__main__":
    # Experiment 1 — k=5 similarity
    print("\n🧪 Running Experiment 1: k=5 similarity search")
    run_experiment(
        TEST_SET,
        config_name="experiment_k5",
        k=5,
        search_type="similarity"
    )

    # Experiment 2 — k=3 MMR
    print("\n🧪 Running Experiment 2: k=3 MMR search")
    run_experiment(
        TEST_SET,
        config_name="experiment_mmr",
        k=3,
        search_type="mmr"
    )

    # Print final comparison
    print_comparison()