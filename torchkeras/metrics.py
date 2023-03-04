# -*- coding: utf-8 -*-
import torch 
from torch import nn 
import sys 

class Accuracy(nn.Module):
    'Accuracy for binary-classification task'
    def __init__(self):
        super().__init__()
        self.correct = nn.Parameter(torch.tensor(0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0),requires_grad=False)
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        correct_i = torch.sum((torch.sigmoid(preds)>=0.5)==(targets>0.5)) 
        total_i = targets.numel()

        self.correct += correct_i 
        self.total += total_i
        return correct_i.float()/total_i
    
    def compute(self):
        return self.correct.float()/self.total 
    
    def reset(self):
        self.correct-=self.correct
        self.total-=self.total

class Precision(nn.Module):
    'Precision for binary-classification task'
    def __init__(self):
        super().__init__()
        self.true_positive = nn.Parameter(torch.tensor(0),requires_grad=False)
        self.false_positive = nn.Parameter(torch.tensor(0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        tpi = torch.sum((y_pred>=0.5)*(y_true>=0.5))
        fpi = torch.sum((y_pred>=0.5)*(y_true<0.5))
        self.true_positive += tpi
        self.false_positive += fpi
        return torch.true_divide(tpi, tpi+fpi)
    
    def compute(self):
        return torch.true_divide(self.true_positive, (self.true_positive+self.false_positive))
    
    def reset(self):
        self.true_positive-=self.true_positive 
        self.false_positive-=self.false_positive
        

class Recall(nn.Module):
    'Recall for binary-classification task'
    def __init__(self):
        super().__init__()
        self.true_positive = nn.Parameter(torch.tensor(0),requires_grad=False)
        self.total_positive = nn.Parameter(torch.tensor(0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = targets.reshape(-1)
        assert y_pred.shape == y_true.shape
        
        true_positive_i = torch.sum((y_pred>=0.5)*(y_true>=0.5))
        total_positive_i = torch.sum(y_true>=0.5)
        self.true_positive += true_positive_i
        self.total_positive += total_positive_i
        return torch.true_divide(true_positive_i,total_positive_i)

    def compute(self):
        return torch.true_divide(self.true_positive,self.total_positive) 
    
    def reset(self):
        self.true_positive -= self.true_positive
        self.total_positive -= self.total_positive
        

class AUC(nn.Module):
    'approximate AUC calculation for binary-classification task'
    def __init__(self):
        super().__init__()
        self.tp = nn.Parameter(torch.zeros(10001),requires_grad=False)
        self.fp = nn.Parameter(torch.zeros(10001),requires_grad=False)
        
    def eval_auc(self,tp,fp):
        tp_total = torch.sum(tp)
        fp_total = torch.sum(fp)
        length = len(tp)
        tp_reverse = tp[range(length-1,-1,-1)]
        tp_reverse_cum = torch.cumsum(tp_reverse,dim=0)-tp_reverse/2.0
        fp_reverse = fp[range(length-1,-1,-1)]
        
        auc = torch.sum(torch.true_divide(tp_reverse_cum,tp_total)
                        *torch.true_divide(fp_reverse,fp_total))
        return auc
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = (10000*torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)
        
        tpi = self.tp-self.tp
        fpi = self.fp-self.fp
        assert y_pred.shape == y_true.shape
        for i,label in enumerate(y_true):
            if label>=0.5:
                tpi[y_pred[i]]+=1.0
            else:
                fpi[y_pred[i]]+=1.0
        self.tp+=tpi
        self.fp+=fpi
        return self.eval_auc(tpi,fpi)
          
    def compute(self):
        return self.eval_auc(self.tp,self.fp)
    
    def reset(self):
        self.tp-=self.tp
        self.fp-=self.fp

class KS(nn.Module):    
    'approximate KS calculation for binary-classification task'
    def __init__(self):
        super().__init__()
        self.tp = nn.Parameter(torch.zeros(10001),requires_grad=False)
        self.fp = nn.Parameter(torch.zeros(10001),requires_grad=False)
        
    def eval_ks(self,tp,fp):
        tp_cum = torch.cumsum(tp,dim = 0)
        fp_cum = torch.cumsum(fp,dim = 0)
        tp_curve = tp_cum/tp_cum[-1]
        fp_curve = fp_cum/fp_cum[-1]
        ks_value = torch.max(torch.abs(tp_curve-fp_curve)) 
        return ks_value
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = (10000*torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)
        
        tpi = self.tp-self.tp
        fpi = self.fp-self.fp
        assert y_pred.shape == y_true.shape
        for i,label in enumerate(y_true):
            if label>=0.5:
                tpi[y_pred[i]]+=1.0
            else:
                fpi[y_pred[i]]+=1.0
                
        self.tp+=tpi
        self.fp+=fpi
        
        return self.eval_ks(tpi,fpi)
    
    def compute(self):
        return self.eval_ks(self.tp,self.fp)
    
    def reset(self):
        self.tp-=self.tp
        self.fp-=self.fp
    

class IOU(nn.Module):
    'IOU calculation for segmentation task (bath binary and multiclass)'
    def __init__(self, num_classes, if_print=False):
        super().__init__()
        self.num_classes = num_classes   
        n = num_classes if num_classes>=2 else 2
        self.mat = nn.Parameter(torch.zeros((n, n), dtype=torch.int64),requires_grad=False)
        self.if_print = if_print

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        n = self.num_classes if self.num_classes>=2 else 2
        with torch.no_grad():
            if self.num_classes>=2:
                a,b = targets.flatten(),preds.argmax(1).flatten()
            else:
                a,b = targets.flatten(),(preds>0).long().flatten()
            assert a.shape == b.shape
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            mati = torch.bincount(inds, minlength=n**2).reshape(n, n)
            self.mat += mati
            acc_global,  iou = self.eval_iou(mati) 
            return iou.mean()

    def compute(self):
        acc_global,  iou = self.eval_iou(self.mat) 
        if self.if_print:
            print(self,file = sys.stderr)
        return iou.mean()
        
    def reset(self):
        self.mat.zero_()
        
    def eval_iou(self,mat):
        h = mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, iou 

    def __str__(self):
        acc_global,iou = self.eval_iou(self.mat)
        return (
            'global correct: {:.4f}\n'
            'IoU: {}\n'
            'mean IoU: {:.4f}').format(
                acc_global.item(),
                ['{:.4f}'.format(i) for i in iou.tolist()],
                iou.mean().item())
