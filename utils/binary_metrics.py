from torcheval.metrics.functional import binary_auprc, binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_auroc
from dataclasses import dataclass
import torch
from typing import List, Dict

@dataclass
class MetricResults:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0 
    
    def measure(self, scores: torch.Tensor | None, pred_labels: torch.Tensor | None, labels: torch.Tensor | None, avoid_curves: bool = False, avoid_thresholds: bool = False):
        if not avoid_thresholds and pred_labels is not None and labels is not None:
            self.accuracy = binary_accuracy(pred_labels, labels).item()
            self.precision = binary_precision(pred_labels, labels).item()
            self.recall = binary_recall(pred_labels, labels).item()
            self.f1_score = binary_f1_score(pred_labels, labels).item()
        
        if not avoid_curves and scores is not None and labels is not None:
            self.auroc = binary_auroc(scores.float(), labels.int()).item()
            self.auprc =  binary_auprc(scores.float(), labels.int()).item()
        
    def get_dict(self, avoid_curves: bool = False, avoid_thresholds: bool = False):
        if avoid_curves and avoid_thresholds:
            raise ValueError("Cannot avoid both curves and thresholds. Please choose one or neither.")
        
        ret_dict: Dict[str, float] = {}
    
        if not avoid_thresholds:
            ret_dict = ret_dict | {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
            }
            
        if not avoid_curves:
            ret_dict = ret_dict | {
            "AUROC": self.auroc,
            "AUPRC": self.auprc
            }
            
        return ret_dict
        
    def __repr__(self):
        return ', '.join([f"{key}: {value:.4f}" for key, value in self.get_dict().items()])

class Metrics():
    def __init__(self, avoid_curves: bool = False, avoid_thresholds: bool = False):
        self.avoid_curves = avoid_curves
        self.avoid_thresholds = avoid_thresholds
        self._scores: List[torch.Tensor] = []
        self._pred_labels: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def reinit(self):
        self._scores = []
        self._pred_labels = []
        self._labels = []

    def update(self, scores: torch.Tensor | None = None, pred_labels: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> None:
        if scores is not None:
            self._scores.append(scores)
        if pred_labels is not None:
            self._pred_labels.append(pred_labels)
        if labels is not None:
            self._labels.append(labels)
    
    def compute(self) -> MetricResults:
        result = MetricResults()
        result.measure(
            scores = torch.cat(self._scores) if self._scores else None,
            pred_labels = torch.cat(self._pred_labels) if self._pred_labels else None,
            labels = torch.cat(self._labels) if self._labels else None,
            avoid_curves=self.avoid_curves,
            avoid_thresholds=self.avoid_thresholds
        )        
        return result
    


if __name__ == "__main__":

    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    pred_labels = torch.tensor([0, 0, 0, 1])
    labels = torch.tensor([0, 1, 0, 1])

    
    metrics = Metrics(avoid_curves=True)
    metrics.update(scores, pred_labels, labels)
    average_results = metrics.compute()
    print(average_results)