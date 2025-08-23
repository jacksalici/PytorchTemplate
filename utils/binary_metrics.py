"""
Utility classes and functions for computing binary classification metrics.

The following metrics are currently supported:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Area Under the Receiver Operating Characteristic Curve (AUROC)
    - Area Under the Precision-Recall Curve (AUPRC)
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    binary_precision,
    binary_recall,
)


@dataclass
class MetricResults:
    """
    Dataclass to store and compute binary classification metrics.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0

    def measure(
        self,
        scores: torch.Tensor | None,
        pred_labels: torch.Tensor | None,
        labels: torch.Tensor | None,
        avoid_curves: bool = False,
        avoid_thresholds: bool = False,
    ) -> None:
        """
        Compute the binary classification metrics.

        Args:
            scores: Tensor containing the predicted scores or probabilities.
                - shape (N_samples,)
            pred_labels: Tensor of shape (N,) with the predicted binary labels
                - shape (N_samples,)
                - binary values (0 or 1)
            labels: Tensor of shape (N,) with the ground truth binary labels
                - shape (N_samples,)
                - binary values (0 or 1)
            avoid_curves: If True, skip computation of AUROC and AUPRC.
            avoid_thresholds: If True, skip computation of threshold-based
                metrics (accuracy, precision, recall, F1 score).

        Raises:
            ValueError: If required inputs for the selected metrics are not
            provided (for example, if threshold-based metrics are to be computed
            but `pred_labels` or `labels` is None).
        """
        if not avoid_thresholds and pred_labels is not None and labels is not None:
            self.accuracy = binary_accuracy(pred_labels, labels).item()
            self.precision = binary_precision(pred_labels, labels).item()
            self.recall = binary_recall(pred_labels, labels).item()
            self.f1_score = binary_f1_score(pred_labels, labels).item()
        elif not avoid_thresholds and (pred_labels is None or labels is None):
            raise ValueError(
                "pred_labels and labels must be provided "
                "to compute threshold-based metrics."
            )

        if not avoid_curves and scores is not None and labels is not None:
            self.auroc = binary_auroc(scores.float(), labels.int()).item()
            self.auprc = binary_auprc(scores.float(), labels.int()).item()
        elif not avoid_curves and (scores is None or labels is None):
            raise ValueError(
                "scores and labels must be provided to compute AUROC and AUPRC."
            )

    def get_dict(
        self, avoid_curves: bool = False, avoid_thresholds: bool = False
    ) -> Dict[str, float]:
        """
        Get the computed metrics as a dictionary.

        Args:
            avoid_curves: If True, exclude AUROC and AUPRC from the output.
            avoid_thresholds: If True, exclude threshold-based metrics
                (accuracy, precision, recall, F1 score) from the output.

        Returns:
            Dictionary with metric names (string) as keys
            and their computed values as floats.

        Raises:
            ValueError: If both `avoid_curves` and `avoid_thresholds` are True.
        """
        if avoid_curves and avoid_thresholds:
            raise ValueError(
                "Cannot avoid both curves and thresholds. Please choose one or neither."
            )

        ret_dict: Dict[str, float] = {}

        if not avoid_thresholds:
            ret_dict = ret_dict | {
                "Accuracy": self.accuracy,
                "Precision": self.precision,
                "Recall": self.recall,
                "F1 Score": self.f1_score,
            }

        if not avoid_curves:
            ret_dict = ret_dict | {"AUROC": self.auroc, "AUPRC": self.auprc}

        return ret_dict

    def __repr__(self):
        return ", ".join(
            [f"{key}: {value:.4f}" for key, value in self.get_dict().items()]
        )


class Metrics:
    """
    Class to accumulate predictions and labels over multiple batches.

    This class allows for the computation of binary classification metrics
    over an entire set by accumulating scores, predicted labels, and true labels.
    """

    def __init__(self, avoid_curves: bool = False, avoid_thresholds: bool = False):
        """
        Args:
            avoid_curves: If True, skip computation of AUROC and AUPRC.
            avoid_thresholds: If True, skip computation of threshold-based
                metrics (accuracy, precision, recall, F1 score).
        """
        self.avoid_curves = avoid_curves
        self.avoid_thresholds = avoid_thresholds
        self._scores: List[torch.Tensor] = []
        self._pred_labels: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def reinit(self) -> None:
        """
        Clear all internal lists storing scores and labels.
        """
        self._scores = []
        self._pred_labels = []
        self._labels = []

    def update(
        self,
        scores: torch.Tensor | None = None,
        pred_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        """
        Accumulate scores and labels from a new batch.

        Args:
            scores: Tensor containing the predicted scores or probabilities.
                - shape (N_samples,)
            pred_labels: Tensor of shape (N,) with the predicted binary labels
                - shape (N_samples,)
                - binary values (0 or 1)
            labels: Tensor of shape (N,) with the ground truth binary labels
                - shape (N_samples,)
                - binary values (0 or 1)
        """
        if scores is not None:
            self._scores.append(scores)
        if pred_labels is not None:
            self._pred_labels.append(pred_labels)
        if labels is not None:
            self._labels.append(labels)

    def compute(self) -> MetricResults:
        """
        Compute and return the accumulated metrics as a MetricResults object.

        Returns:
            MetricResults object containing the computed metrics.
        """
        result = MetricResults()
        result.measure(
            scores=torch.cat(self._scores) if self._scores else None,
            pred_labels=torch.cat(self._pred_labels) if self._pred_labels else None,
            labels=torch.cat(self._labels) if self._labels else None,
            avoid_curves=self.avoid_curves,
            avoid_thresholds=self.avoid_thresholds,
        )
        return result


def demo() -> None:
    """
    Quick demo of the Metrics class.
    """
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    pred_labels = torch.tensor([0, 0, 0, 1])
    labels = torch.tensor([0, 1, 0, 1])

    metrics = Metrics(avoid_curves=True)
    metrics.update(scores, pred_labels, labels)
    average_results = metrics.compute()
    print(average_results)


if __name__ == "__main__":
    demo()
