import torch
import torch.nn as nn
from functools import partial

from utils.calc_acc import calc_acc
from models.resnet import resnet50
from layers.separate_bn import SeparateBatchNorm
from losses.triplet_loss import TripletLoss


class Baseline(nn.Module):
    def __init__(self, num_classes=None, **kwargs):
        super(Baseline, self).__init__()
        self.backbone = resnet50(pretrained=True, last_stride=1)

        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)
      

        self._eval_mode = 'visible'

        if num_classes is not None:
            self.classifier = nn.Linear(2048, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

            # losses
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.triplet_loss = TripletLoss(margin=0.3)

    @property
    def eval_mode(self):
        return self._eval_mode

    @eval_mode.setter
    def eval_mode(self, mode):
        if mode not in ('visible', 'infrared'):
            raise ValueError('The choice of mode is visible or infrared!')
        self._eval_mode = mode

        def set_sep_bn_mode(m, eval_mode):
            if isinstance(m, SeparateBatchNorm):
                m.eval_mode = eval_mode

        set_sep_bn_mode = partial(set_sep_bn_mode, eval_mode=mode)
        self.backbone.apply(set_sep_bn_mode)

    def get_param_groups(self, lr, weight_decay):
        ft_params = self.backbone.parameters()
        new_params = [param for name, param in self.named_parameters() if not name.startswith("backbone.")]
        param_groups = [{'params': ft_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
                        {'params': new_params, 'lr': lr, 'weight_decay': weight_decay}]
        return param_groups

    def forward(self, inputs, labels=None, **kwargs):
        global_feat = self.backbone(inputs)
        if self.training:
            return self.train_forward(global_feat, labels, **kwargs)
        return self.test_forward(global_feat)

    def test_forward(self, feats):
        return self.bn_neck(feats)

    def train_forward(self, feats, labels, **kwargs):
        triplet_loss = self.triplet_loss(feats, labels)

        logits = self.classifier(self.bn_neck(feats))
        cls_loss = self.id_loss(logits, labels)

        loss = triplet_loss + cls_loss 
        metrics = {'ce': cls_loss.item(), 'acc': calc_acc(logits.data, labels.data), 'tri': triplet_loss.item()}

        return loss, metrics
