import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def convert_sep_bn_model(module, separate_affine=False):
    mod = module

    if isinstance(module, nn.BatchNorm2d):
        mod = SeparateBatchNorm(num_features=module.num_features, separate_affine=separate_affine,
                                momentum=module.momentum, affine=module.affine, eps=module.eps,
                                track_running_stats=module.track_running_stats)

        mod.running_mean = module.running_mean.clone()
        mod.running_var = module.running_var.clone()
        mod.running_mean_s = module.running_mean.clone()
        mod.running_var_s = module.running_var.clone()

        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            if separate_affine:
                mod.weight_s.data = module.bias.data.clone().detach()
                mod.bias_s.data = module.bias.data.clone().detach()

    for name, children in module.named_children():
        mod.add_module(name, convert_sep_bn_model(children, separate_affine))

    del module
    return mod


class SeparateBatchNorm(_BatchNorm):
    def __init__(self, num_features, separate_affine=False, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(SeparateBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('running_mean_s', torch.zeros(num_features))
        self.register_buffer('running_var_s', torch.ones(num_features))

        if affine and separate_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight_s = self.weight
            self.bias_s = self.bias

        self._eval_mode = 'visible'

    @property
    def eval_mode(self):
        return self._eval_mode

    @eval_mode.setter
    def eval_mode(self, mode):
        if mode not in ('visible', 'infrared'):
            raise ValueError('The choice of mode is visible or infrared!')
        self._eval_mode = mode

    def forward(self, x):
        if self.training:
            source_split, target_split = x.tensor_split(2, dim=0)
            source_result = F.batch_norm(source_split, self.running_mean_s, self.running_var_s, self.weight_s,
                                         self.bias_s, True, self.momentum, self.eps)
            target_result = F.batch_norm(target_split, self.running_mean, self.running_var, self.weight,
                                         self.bias, True, self.momentum, self.eps)

            return torch.cat([source_result, target_result], dim=0)
        else:
            if self.eval_mode == 'infrared':
                result = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                      not self.track_running_stats, self.momentum, self.eps)
            else:
                result = F.batch_norm(x, self.running_mean_s, self.running_var_s, self.weight_s, self.bias_s,
                                      not self.track_running_stats, self.momentum, self.eps)

            return result
