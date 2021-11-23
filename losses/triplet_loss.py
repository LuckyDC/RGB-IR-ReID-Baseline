import torch
from torch import nn
import torch.nn.functional as F


def hard_example_mining(dist_mat, labels_1, labels_2=None):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels_1: pytorch LongTensor, with shape [N]
      labels_2: pytorch LongTensor, with shape [N]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    if labels_2 is None:
        labels_2 = labels_1

    assert dist_mat.dim() == 2
    assert dist_mat.size(0) == labels_1.size(0)
    assert dist_mat.size(1) == labels_2.size(0)
    m, n = dist_mat.shape

    labels_1 = labels_1.view(-1, 1).expand(-1, n)
    labels_2 = labels_2.view(1, -1).expand(m, -1)
    pos_mask = labels_1.eq(labels_2).to(dtype=dist_mat.dtype)
    neg_mask = 1 - pos_mask

    dist_ap, _ = torch.max(dist_mat - neg_mask * 1e4, dim=1)
    dist_an, _ = torch.min(dist_mat + pos_mask * 1e4, dim=1)

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    def __init__(self, margin=None, normalize=False, reduction='mean'):
        super(TripletLoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none'], 'reduction = "{}" is not supported.'.format(reduction)

        self.margin = margin
        self.normalize = normalize
        self.reduction = reduction
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction='none')

    def forward(self, feat_1, labels_1, feat_2=None, labels_2=None):
        if feat_2 is None:
            feat_2 = feat_1
            labels_2 = labels_1

        if self.normalize:
            feat_1 = F.normalize(feat_1, p=2, dim=-1)
            feat_2 = F.normalize(feat_2, p=2, dim=-1)

        dist_mat = torch.cdist(feat_1, feat_2, p=2)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels_1, labels_2)
        y = torch.ones_like(dist_ap, dtype=torch.long)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
