import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from ..builder import LOSSES
from .utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def wasserstein_dis(pred,
                    soft_label,
                    g=1.0):
    hot_ind=torch.argmax(soft_label,dim=1).cuda()
    position_array=torch.abs(
        torch.arange(0,pred.size()[1]).expand(pred.size()[0],pred.size()[1]).cuda()-hot_ind.reshape(soft_label.size()[0],-1).expand(soft_label.size()[0],soft_label.size()[1])
    )**g
    max_dis=(torch.abs(
        pred.size()[1] / 2 - (1+hot_ind)
    )+pred.size()[1] / 2)**g
    '''
    if(0):
        print(position_array)
        print(pred)
        print(position_array*pred)
    '''
    #print((position_array*pred).sum(axis=1)/max_dis)
    return (position_array*pred).sum(axis=1) #/max_dis




@LOSSES.register_module()
class SinkhornDistanceOneHot(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
        Tmode: usage of T
            Tmode=
                0 -> x1(pred) y1(label)
                1 -> x1
                2 -> y1
                else -> not use T
        g (float): gamma
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, reduction='mean', loss_weight=1.0,T=1,Tmode=1,g=1.0):
        super(SinkhornDistanceOneHot, self).__init__()
        self.g = g
        self.reduction = reduction
        self.loss_weight=loss_weight
        self.T=T
        self.Tmode=Tmode

    def forward(self, x1, y1,weight=None, avg_factor=None,
                reduction_override=None):
        if (self.Tmode==0):
            x1 = F.softmax(x1 / self.T,dim=1)
            y1 = F.softmax(y1 / self.T, dim=1)
        elif (self.Tmode==1):
            x1 = F.softmax(x1 / self.T,dim=1)
            #y1 = F.softmax(y1 / self.T, dim=1)
        elif (self.Tmode==2):
            #x1 = F.softmax(x1 / self.T,dim=1)
            y1 = F.softmax(y1 / self.T, dim=1)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        return wasserstein_dis(x1,
                               y1,
                               weight=weight,
                               reduction=reduction,
                               avg_factor=avg_factor,
                               g=self.g)
