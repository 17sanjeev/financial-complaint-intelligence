"""
Evaluation utilities for complaint classification and triage.
Author: Sanjeev Kumar | IIT Bombay
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, classification_report,
    average_precision_score, roc_auc_score,
    brier_score_loss, confusion_matrix
)


def classification_metrics(y_true, y_pred, name="Model"):
    """Product classification metrics."""
    macro_f1    = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{name}:")
    print(f"  Macro-F1:    {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred))
    return macro_f1, weighted_f1


def triage_metrics(y_true, y_prob, name="Triage Model"):
    """
    Business-oriented triage metrics.
    DO NOT use accuracy for 1% positive rate!
    Use PR-AUC, Recall@K, Precision@K, Lift@K
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)
    baseline_pr = y_true.mean()

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  PR-AUC (main):  {pr_auc:.4f}  (random={baseline_pr:.4f})")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  Brier Score:    {brier:.4f}")
    print(f"  PR-AUC Lift:    {pr_auc/baseline_pr:.1f}x over random")

    n = len(y_true)
    sorted_idx = np.argsort(y_prob)[::-1]
    results = {}

    print(f"\n  Business Metrics (Review Queue Analysis):")
    for pct in [0.05, 0.10, 0.20]:
        k = int(n * pct)
        top_k_true  = y_true[sorted_idx[:k]]
        recall_k    = top_k_true.sum() / y_true.sum()
        precision_k = top_k_true.mean()
        lift_k      = recall_k / pct

        print(f"\n  Top {pct:.0%} review queue ({k:,} complaints):")
        print(f"    Recall@{pct:.0%}:    {recall_k:.1%} of all priority complaints captured")
        print(f"    Precision@{pct:.0%}: {precision_k:.1%} of reviewed are priority")
        print(f"    Lift@{pct:.0%}:      {lift_k:.1f}x over random review")

        results[f'recall_top_{int(pct*100)}pct'] = recall_k
        results[f'precision_top_{int(pct*100)}pct'] = precision_k
        results[f'lift_top_{int(pct*100)}pct'] = lift_k

    results['pr_auc']  = pr_auc
    results['roc_auc'] = roc_auc
    results['brier']   = brier
    return results


def print_model_comparison(results_dict):
    """Print final model comparison table."""
    print("\n" + "="*65)
    print("FINAL MODEL COMPARISON TABLE")
    print("="*65)
    print(f"{'Model':<35} {'Val F1':>8} {'Test F1':>10} {'Note'}")
    print("-"*65)
    for name, r in results_dict.items():
        note = r.get('note', '')
        print(f"{name:<35} {r.get('val_f1', 0):>8.4f} {r.get('test_f1', 0):>10.4f}  {note}")
