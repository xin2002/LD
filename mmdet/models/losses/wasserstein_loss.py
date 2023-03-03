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
                    max_iter,
                    eps):
    C = _cost_matrix(pred, soft_label)  # Wasserstein cost function
    x_points = pred.shape[-2]
    y_points = soft_label.shape[-2]
    if pred.dim() == 2:
        batch_size = 1
    else:
        batch_size = pred.shape[0]

    # both marginals are fixed with equal weights
    mu = torch.empty(batch_size, x_points, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / x_points).squeeze()
    nu = torch.empty(batch_size, y_points, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / y_points).squeeze()
    mu = mu.cuda()
    nu = nu.cuda()
    u = torch.zeros_like(mu).cuda()
    v = torch.zeros_like(nu).cuda()

    # To check if algorithm terminates because of threshold
    # or max iterations reached
    actual_nits = 0
    # Stopping criterion
    thresh = 1e-1
    # Sinkhorn iterations
    for i in range(max_iter):
        u1 = u  # useful to check the update
        u = eps * (torch.log(mu + 1e-8) - torch.logsumexp(M(eps,C, u, v), dim=-1)) + u
        v = eps * (torch.log(nu + 1e-8) - torch.logsumexp(M(eps,C, u, v).transpose(-2, -1), dim=-1)) + v
        err = (u - u1).abs().sum(-1).mean()

        actual_nits += 1
        if err.item() < thresh:
            break
    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    pi = torch.exp(M(eps,C, U, V)).cuda()
    # Sinkhorn distance
    cost = torch.sum(pi * C, dim=(-2, -1))

    return cost

def M(eps, C, u, v):
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
    return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps


def _cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    # print(x_col)
    y_lin = y.unsqueeze(-3)
    # print(y_lin)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C


def ave(u, u1, tau):
    "Barycenter subroutine, used by kinetic acceleration through extrapolation."
    return tau * u + (1 - tau) * u1


@LOSSES.register_module()
class SinkhornDistance(nn.Module):
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
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=0.1, max_iter=30, reduction='mean', loss_weight=1.0,T=10,Tmode=0):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.loss_weight=loss_weight
        self.T=T
        self.Tmode=Tmode

    def forward(self, x1, y1,weight=None, avg_factor=None,
                reduction_override=None):
        grad_coef=1
        if (self.Tmode==0):
            x1 = F.softmax(x1 / self.T,dim=1)
            y1 = F.softmax(y1 / self.T, dim=1)
            grad_coef = self.T*self.T
        elif (self.Tmode==1):
            x1 = F.softmax(x1 / self.T,dim=1)
            #y1 = F.softmax(y1 / self.T, dim=1)
            grad_coef = self.T
        elif (self.Tmode==2):
            #x1 = F.softmax(x1 / self.T,dim=1)
            y1 = F.softmax(y1 / self.T, dim=1)
            grad_coef = self.T
        x1 = torch.unsqueeze(x1,2).cuda()
        y1 = torch.unsqueeze(y1,2).cuda()
        x3 = (torch.unsqueeze(torch.arange(float(x1.size()[1])),1)* torch.ones(x1.size()[0], x1.size()[1], 1)).cuda()
        y3 = (torch.unsqueeze(torch.arange(float(y1.size()[1])),1)* torch.ones(y1.size()[0], y1.size()[1], 1)).cuda()
        x = torch.cat((x3, x1), dim=2).cuda()
        y = torch.cat((y3, y1), dim=2).cuda()
        #print(x)
        #print(y)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        return wasserstein_dis(x,
                               y,
                               weight=weight,
                               reduction=reduction,
                               avg_factor=avg_factor,
                               max_iter=self.max_iter,
                               eps=self.eps,)*grad_coef 

