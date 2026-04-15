"""
Regression gate thresholds for PaperMind RAGAS eval.

A run is considered passing only when ALL metrics meet or exceed their target.
Update these only when a deliberate architectural decision justifies the change —
not to make a failing run pass.
"""

THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_recall":    0.80,
    "context_precision": 0.70,
}
