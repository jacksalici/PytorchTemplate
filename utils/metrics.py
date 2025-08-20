from torcheval.metrics.functional import binary_auprc, binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_auroc
from dataclasses import dataclass
import torch
from typing import List

@dataclass
class MetricResults:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0 
    
    def measure(self, scores, pred_labels, labels, adjust=False, avoid_curves = False, avoid_thresholds = False):
        if adjust:
            from utils.tools import adjustment
            labels, pred_labels = adjustment(labels, pred_labels)
        
        if not avoid_thresholds:
            self.accuracy = binary_accuracy(pred_labels, labels)
            self.precision = binary_precision(pred_labels, labels)
            self.recall = binary_recall(pred_labels, labels)
            self.f1_score = binary_f1_score(pred_labels, labels)
        
        if not avoid_curves:
            self.auroc = binary_auroc(scores.float(), labels.int())
            self.auprc =  binary_auprc(scores.float(), labels.int())
        
    def get_dict(self, avoid_curves = False, avoid_thresholds = False):
        if avoid_curves and avoid_thresholds:
            raise ValueError("Cannot avoid both curves and thresholds. Please choose one or neither.")
        
        ret_dict = {}
    
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
    def __init__(self, avoid_curves = False, avoid_thresholds = False):
        self.avoid_curves = avoid_curves
        self.avoid_thresholds = avoid_thresholds
        self._scores: List[torch.Tensor] = []
        self._pred_labels: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def reinit(self):
        self._scores = []
        self._pred_labels = []
        self._labels = []

    def update(self, scores = None, pred_labels = None, labels = None, adjust=False):
        if scores is not None:
            self._scores.append(scores)
        if pred_labels is not None:
            self._pred_labels.append(pred_labels)
        if labels is not None:
            self._labels.append(labels)
    
    def compute(self):
        result = MetricResults()
        result.measure(
            scores = torch.cat(self._scores) if self._scores else None,
            pred_labels = torch.cat(self._pred_labels) if self._pred_labels else None,
            labels = torch.cat(self._labels) if self._labels else None,
            adjust=False,
            avoid_curves=self.avoid_curves,
            avoid_thresholds=self.avoid_thresholds
        )        
        return result
    


if __name__ == "__main__":

    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    pred_labels = torch.tensor([0, 0, 0, 1])
    labels = torch.tensor([0, 1, 0, 1])

    
    metrics = Metrics(avoid_curves=True)
    metrics.update(scores, pred_labels, labels, adjust=False)
    average_results = metrics.compute()
    print(average_results)