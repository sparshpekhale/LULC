import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduce=None)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss

def calculate_metrics(y_true, y_pred, num_classes):
    # Flatten both true and predicted labels
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Initialize variables to store metrics
    accuracy = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    precision = torch.zeros(num_classes)
    iou = torch.zeros(num_classes)

    for cls in range(num_classes):
        # True positives
        TP = torch.sum((y_true_flat == cls) & (y_pred_flat == cls)).item()
        
        # False positives
        FP = torch.sum((y_true_flat != cls) & (y_pred_flat == cls)).item()
        
        # False negatives
        FN = torch.sum((y_true_flat == cls) & (y_pred_flat != cls)).item()
        
        # Intersection
        intersection = torch.sum((y_true_flat == cls) & (y_pred_flat == cls)).item()
        
        # Union
        union = TP + FP + FN
        
        # Calculate metrics
        if union != 0:
            accuracy[cls] = intersection / union
            recall[cls] = TP / (TP + FN) if (TP + FN) != 0 else 0
            precision[cls] = TP / (TP + FP)
            iou[cls] = intersection / (union - intersection)

    # Overall metrics
    overall_accuracy = torch.mean(accuracy)
    overall_recall = torch.mean(recall)
    overall_precision = torch.mean(precision)
    overall_iou = torch.mean(iou)

    return overall_accuracy, overall_recall, overall_precision, overall_iou, accuracy, recall, precision, iou

def calculate_class_metrics(y_true, y_pred, cls):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    TP = torch.sum((y_true_flat == cls) & (y_pred_flat == cls)).item()    
    FP = torch.sum((y_true_flat != cls) & (y_pred_flat == cls)).item()
    FN = torch.sum((y_true_flat == cls) & (y_pred_flat != cls)).item()

    intersection = torch.sum((y_true_flat == cls) & (y_pred_flat == cls)).item()
    union = TP + FP + FN
    
    # Calculate metrics
    if union != 0:
        accuracy = intersection / union
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP)
        iou = intersection / (union - intersection)
    else:
        return [0 for i in range(4)]

    return accuracy, recall, precision, iou