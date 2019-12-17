import torch.nn as nn
import numpy as np

from models.resnet import resnet50
from utils.calc_acc import calc_acc

from layers import TripletLoss


class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, dual_path=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.dual_path = dual_path

        if self.dual_path:
            self.backbone_1 = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
            self.backbone_2 = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        else:
            self.backbone_1 = self.backbone_2 = resnet50(pretrained=True, drop_last_stride=drop_last_stride)

        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)

        if self.classification:
            self.classifier = nn.Linear(2048, num_classes, bias=False)
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=0.3)

    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        modal_flag_1 = np.in1d(cam_ids.cpu().numpy(), [1, 2, 4, 5])
        modal_flag_2 = np.logical_not(modal_flag_1)

        if not self.dual_path:
            global_feat = self.backbone_1(inputs)
        else:
            if np.any(modal_flag_1) and np.any(modal_flag_2):
                global_feat_1 = self.backbone_1(inputs[modal_flag_1])
                global_feat_2 = self.backbone_2(inputs[modal_flag_2])

                global_feat = global_feat_1.new_empty(size=(inputs.size(0),) + global_feat_1.size()[1:])
                global_feat[modal_flag_1, ...] = global_feat_1
                global_feat[modal_flag_2, ...] = global_feat_2
            elif np.any(modal_flag_1):
                global_feat = self.backbone_1(inputs[modal_flag_1])
            else:
                global_feat = self.backbone_2(inputs[modal_flag_2])

        if not self.training:
            global_feat = self.bn_neck(global_feat)
            return global_feat
        else:
            return self.train_forward(global_feat, labels, **kwargs)

    def train_forward(self, inputs, labels, **kwargs):
        loss = 0
        metric = {}

        if self.triplet:
            triplet_loss, _, _ = self.triplet_loss(inputs.float(), labels)
            loss += triplet_loss
            metric.update({'tri': triplet_loss.data})

        inputs = self.bn_neck(inputs)

        if self.classification:
            logits = self.classifier(inputs)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        return loss, metric
