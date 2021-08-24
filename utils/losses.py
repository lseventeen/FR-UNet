import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        # target_flat = target.contiguous().view(-1)
        # intersection = (output_flat * target_flat).sum()
        # loss = 1 - ((2. * intersection + self.smooth) /
        #             (output_flat.sum() + target_flat.sum() + self.smooth))
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss

class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)



class CE_Loss3(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_Loss3, self).__init__()
        
        self.l_Loss = BCELoss(reduction=reduction)
        self.s_Loss = BCELoss(reduction=reduction)
        self.f_Loss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, pre,l_pre,s_pre, tar,l_tar,s_tar):
        return self.D_weight * self.f_Loss(pre, tar) + (1 - self.D_weight)/2 * self.l_Loss(l_pre,l_tar)+(1 - self.D_weight)/2 * self.s_Loss(s_pre,s_tar)