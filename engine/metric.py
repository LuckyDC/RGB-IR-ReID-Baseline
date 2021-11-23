import torch
import torch.distributed as dist
from collections import defaultdict
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced


class AutoKVMetric(Metric):
    def __init__(self):
        self.kv_sum_metric = defaultdict(lambda: torch.tensor(0., device="cuda"))
        self.kv_sum_inst = defaultdict(lambda: torch.tensor(0., device="cuda"))
        self.kv_metric = defaultdict(lambda: 0)

        self.reset()

    @reinit__is_reduced
    def update(self, output):
        if not isinstance(output, dict):
            raise TypeError('The output must be a key-value dict.')

        for k in output.keys():
            self.kv_sum_metric[k].add_(output[k])
            self.kv_sum_inst[k].add_(1)

    @reinit__is_reduced
    def reset(self):
        self.kv_sum_metric.clear()
        self.kv_sum_inst.clear()
        self.kv_metric.clear()

    def compute(self):
        for k in self.kv_sum_metric.keys():
            if self.kv_sum_inst[k] == 0:
                raise NotComputableError('Accuracy must have at least one example before it can be computed')

            metric_value = self.kv_sum_metric[k] / self.kv_sum_inst[k]

            if dist.is_initialized():
                dist.barrier()
                dist.all_reduce(metric_value)
                dist.barrier()
                metric_value /= dist.get_world_size()

            self.kv_metric[k] = metric_value.item()

        return self.kv_metric
