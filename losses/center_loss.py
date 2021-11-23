import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feature_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feature_dim, reduction='mean'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.reduction = reduction
        self.centers = nn.Parameter(torch.Tensor(self.num_classes, self.feature_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers)

    @torch.no_grad()
    def ema_update(self, feats, labels, momentum=0.0):
        unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)

        feat_list = []
        for i in range(unique_labels.size(0)):
            mask = inverse_indices.eq(i)
            feat_list.append(feats[mask].mean(dim=0))

        feats = torch.stack(feat_list, dim=0).to(dtype=self.centers.dtype)
        self.centers.data[unique_labels] *= momentum
        self.centers.data[unique_labels] += (1 - momentum) * feats

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        dist_mat = torch.cdist(x, self.centers, p=2) ** 2 / 2

        classes = torch.arange(self.num_classes, device=x.device, dtype=torch.long)
        classes = classes.unsqueeze(0).expand(x.size(0), -1)
        labels = labels.unsqueeze(1).expand(-1, self.num_classes)
        mask = labels.eq(classes)

        dist = dist_mat * mask.float()
        loss = dist.sum(dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
