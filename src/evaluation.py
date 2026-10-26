"""
evaluation.py
-------------
Stage 8: how good is the detector, really?

This file is what separates "I found some duplicates" from "I measured how
well my method performs". Without it, every number in the dashboard is an
opinion.

We compare our predictions against data/ground_truth.csv, which lists the
pairs we KNOW are the same person (because make_dataset.py built them that
way and then hid the entity_id).

The four outcomes, for one pair:

                      truth: duplicate     truth: different
    we say duplicate   True Positive (TP)   False Positive (FP)
    we say different   False Negative (FN)  True Negative (TN)

    FP = we merged two different customers.  Usually the expensive mistake:
         data is destroyed and the customer sees someone else's record.
    FN = we missed a duplicate. Annoying, but the record is still there.

    Precision = TP / (TP + FP)   of the pairs we flagged, how many were right?
    Recall    = TP / (TP + FN)   of the real duplicates, how many did we find?
    F1        = 2PR / (P + R)    harmonic mean - punishes an unbalanced pair

Run:  python -m src.evaluation
"""

from pathlib import Path

import pandas as pd

from src.matching import DEFAULT_THRESHOLDS, RESULT_COLUMNS, run_pipeline
from src.preprocessing import load_data

SWEEP_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_ground_truth(path="data/ground_truth.csv"):
    """
    Returns (true_pairs, labelled_negatives).

    true_pairs         pairs the data generator knows are the same person
    labelled_negatives pairs we explicitly marked as DIFFERENT people who
                       look similar (same name, same city, sometimes the
                       same address). These are the traps.

    Any pair not listed at all is treated as a non-duplicate. That is the
    standard closed-world assumption in record linkage, and it is safe here
    because we generated the data and know it is complete.
    """
    gt = pd.read_csv(path)
    as_tuples = lambda frame: {
        (r.record_id_1, r.record_id_2) for r in frame.itertuples()
    }
    return (
        as_tuples(gt[gt.is_duplicate == 1]),
        as_tuples(gt[gt.is_duplicate == 0]),
    )


def confusion(results, true_pairs, threshold):
    """
    Count TP / FP / FN / TN at one threshold.

    Two details that are easy to get wrong:

    1. FN is counted against the WHOLE ground truth, not just against the
       candidate pairs. A true duplicate that blocking threw away in Stage 4
       is still a miss - hiding it would let us cheat by blocking harder.

    2. TN is counted over the candidate pairs only, and we also report the
       full-universe figure. The ~103,000 pairs blocking eliminated are all
       true negatives, and including them would inflate accuracy to 99.9%
       no matter how bad the model was.
    """
    predicted = {
        (r.record_1, r.record_2)
        for r in results[results.final_score >= threshold].itertuples()
    }
    candidates = {(r.record_1, r.record_2) for r in results.itertuples()}

    tp = predicted & true_pairs
    fp = predicted - true_pairs
    fn = true_pairs - predicted
    tn = candidates - predicted - true_pairs

    return {
        "threshold": threshold,
        "TP": len(tp), "FP": len(fp), "FN": len(fn), "TN": len(tn),
        "tp_pairs": tp, "fp_pairs": fp, "fn_pairs": fn,
    }


def metrics(counts):
    """Precision, recall, F1 - plus accuracy, to show why we do not use it."""
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    return {
        "threshold": counts["threshold"],
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
    }


def evaluate(results, true_pairs, threshold=None):
    """Metrics at a single threshold."""
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS["duplicate"]
    return metrics(confusion(results, true_pairs, threshold))


def threshold_sweep(results, true_pairs, thresholds=None):
    """
    The precision-recall trade-off, measured.

    Raising the threshold means demanding more evidence before calling a
    pair a duplicate. Fewer pairs clear the bar, so:
        precision goes UP   (the survivors are the obvious ones)
        recall goes DOWN    (the borderline real duplicates get dropped)
    F1 usually peaks somewhere in the middle. There is no universally
    correct threshold - it depends on whether a wrong merge or a missed
    duplicate costs you more.
    """
    rows = [metrics(confusion(results, true_pairs, t))
            for t in (thresholds or SWEEP_THRESHOLDS)]
    return pd.DataFrame(rows)


def confusion_matrix_frame(counts):
    """The 2x2 table, shaped for display."""
    return pd.DataFrame(
        [[counts["TP"], counts["FN"]], [counts["FP"], counts["TN"]]],
        index=["Truth: duplicate", "Truth: different"],
        columns=["Predicted: duplicate", "Predicted: different"],
    )


def error_examples(results, counts, n=5):
    """
    The pairs we got wrong - the most useful output in this whole file.

    A metric tells you THAT the model is wrong. These rows tell you WHY,
    and that is what drives the next improvement.
    """
    key = results.set_index(["record_1", "record_2"])
    fp = key.loc[list(counts["fp_pairs"])] if counts["fp_pairs"] else key.iloc[0:0]
    fn = key.loc[list(counts["fn_pairs"])] if counts["fn_pairs"] else key.iloc[0:0]
    cols = ["comp_name", "comp_email", "comp_phone", "comp_address",
            "fields_compared", "final_score"]
    return (
        fp.sort_values("final_score", ascending=False)[cols].head(n).reset_index(),
        fn.sort_values("final_score", ascending=False)[cols].head(n).reset_index(),
    )


def blocking_note(pairs, true_pairs):
    """How many answers blocking removed before scoring ever ran."""
    kept = len(set(pairs) & true_pairs)
    return {
        "true_pairs": len(true_pairs),
        "survived_blocking": kept,
        "unreachable": len(true_pairs) - kept,
        "recall_ceiling": round(kept / len(true_pairs), 4) if true_pairs else 0.0,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 170)

    out = run_pipeline(load_data("data/sample_data.csv"))
    results = out["results"]
    true_pairs, hard_negatives = load_ground_truth()

    print("--- recall ceiling set by blocking ---")
    for k, v in blocking_note(out["pairs"], true_pairs).items():
        print(f"  {k:<20}: {v}")

    counts = confusion(results, true_pairs, DEFAULT_THRESHOLDS["duplicate"])
    m = metrics(counts)

    print(f"\n--- confusion matrix at threshold {m['threshold']} ---")
    print(confusion_matrix_frame(counts).to_string())

    print("\n--- metrics ---")
    for k in ["precision", "recall", "f1", "accuracy"]:
        print(f"  {k:<10}: {m[k]:.3f}")
    print("\n  Note how useless accuracy is here: predicting 'no duplicates'")
    print(f"  for every pair would score {counts['TN'] / (counts['TN'] + counts['TP'] + counts['FP'] + counts['FN']):.3f}")
    print("  because the classes are heavily imbalanced. Precision and recall")
    print("  ignore TN, which is exactly why we use them.")

    print("\n--- threshold experiment ---")
    sweep = threshold_sweep(results, true_pairs)
    print(sweep[["threshold", "TP", "FP", "FN", "precision", "recall", "f1"]]
          .to_string(index=False))
    best = sweep.loc[sweep.f1.idxmax()]
    print(f"\n  best F1 = {best.f1:.3f} at threshold {best.threshold}")

    fp_df, fn_df = error_examples(results, counts)
    print("\n--- false positives (we merged two different people) ---")
    print(fp_df.to_string(index=False) if len(fp_df) else "  none")

    print("\n--- false negatives (we missed a real duplicate) ---")
    print(fn_df.to_string(index=False) if len(fn_df) else "  none")

    caught = len(counts["fp_pairs"] & hard_negatives)
    print(f"\n  of our {counts['FP']} false positives, {caught} are the "
          f"deliberately confusable pairs from make_dataset.py")

    # Save the artefacts the report needs.
    Path("outputs").mkdir(exist_ok=True)
    results[RESULT_COLUMNS].to_csv("outputs/duplicate_results.csv", index=False)
    sweep.to_csv("outputs/threshold_experiment.csv", index=False)
    confusion_matrix_frame(counts).to_csv("outputs/confusion_matrix.csv")
    print("\n  wrote outputs/duplicate_results.csv, "
          "outputs/threshold_experiment.csv, outputs/confusion_matrix.csv")
