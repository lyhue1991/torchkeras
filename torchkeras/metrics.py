# -*- coding: utf-8 -*-
from torchmetrics import Metric
import torchmetrics
import torch 

"""
Attention!
These metrics below are only for binary-classification task !
for multiclass or regression, please use the metrics in torchmetrics！
torchmetrics need the labels to be torch.long dtype in classification task.
while nn.BCEWithLogitsLoss need the label to be torch.float dtype. 
so we need to use the metrics class below to suit the torch.float labels.
"""

class Accuracy(Metric):
    
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        self.correct += torch.sum((torch.sigmoid(preds)>=0.5)==(targets>0.5)) 
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total 

class Precision(Metric):
    
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        self.true_positive += torch.sum((y_pred>=0.5)*(y_true>=0.5))
        self.false_positive += torch.sum((y_pred>=0.5)*(y_true<0.5))

    def compute(self):
        return torch.true_divide(self.true_positive, (self.true_positive+self.false_positive))

class Recall(Metric):
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_positive", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        self.true_positive += torch.sum((y_pred>=0.5)*(y_true>=0.5))
        self.total_positive += torch.sum(y_true>=0.5)

    def compute(self):
        return torch.true_divide(self.true_positive,self.total_positive) 

class AUC(torchmetrics.AUROC):
    
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        super().update(torch.sigmoid(preds),targets.long())
            
    def compute(self):
        return super().compute()

#AUC近似计算
class AUCROC(Metric):
    
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.zeros(10001), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(10001), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = (10000*torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        for i,label in enumerate(y_true):
            if label>=0.5:
                self.tp[y_pred[i]]+=1.0
            else:
                self.fp[y_pred[i]]+=1.0
            
    def compute(self):
        tp_total = torch.sum(self.tp)
        fp_total = torch.sum(self.fp)
        length = len(self.tp)
        tp_reverse = self.tp[range(length-1,-1,-1)]
        tp_reverse_cum = torch.cumsum(tp_reverse,dim=0)-tp_reverse/2.0
        fp_reverse = self.fp[range(length-1,-1,-1)]

        auc = torch.sum(torch.true_divide(tp_reverse_cum,tp_total)
                        *torch.true_divide(fp_reverse,fp_total))
        return auc

class KS(Metric):
    
    is_differentiable  = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.zeros(10001), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(10001), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = (10000*torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        for i,label in enumerate(y_true):
            if label>=0.5:
                self.tp[y_pred[i]]+=1.0
            else:
                self.fp[y_pred[i]]+=1.0
            
    def compute(self):
        tp_cum = torch.cumsum(self.tp,dim = 0)
        fp_cum = torch.cumsum(self.fp,dim = 0)
        tp_curve = tp_cum/tp_cum[-1]
        fp_curve = fp_cum/fp_cum[-1]
        ks_value = torch.max(torch.abs(tp_curve-fp_curve)) 
        
        return ks_value
