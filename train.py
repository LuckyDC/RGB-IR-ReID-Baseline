import os
import shutil

import torch
from torch import optim
import torch.distributed as dist

from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.baseline import Baseline
from utils.lr_scheduler import WarmupMultiStepLR


def train(cfg):
    # training data loader
    train_loader = get_train_loader(root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    num_workers=8)

    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(root=cfg.data_root,
                                                       batch_size=512,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)

    # model
    model = Baseline(num_classes=cfg.num_id)
    model.cuda()

    # optimizer
    assert cfg.optimizer in ['adam', 'sgd']
    param_groups = model.get_param_groups(lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(param_groups, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.wd)
    else:
        optimizer = optim.SGD(param_groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)

    # convert model for mixed precision training
    lr_scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                     milestones=cfg.lr_step,
                                     gamma=0.1,
                                     warmup_epochs=10,
                                     warmup_factor=0.01)
    # engine
    engine = get_trainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         log_period=cfg.log_period,
                         eval_interval=cfg.eval_interval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader,
                         enable_amp=cfg.fp16)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)


if __name__ == '__main__':
    import yaml
    import time
    import argparse
    import random
    import numpy as np
    from pprint import pformat
    from datetime import timedelta
    from runx.logx import logx
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='Path to config file')
    parser.add_argument('--work-dir', type=str, help='Directory for log and checkpoint')
    parser.add_argument('--gpu', type=int, help='GPU device for training')
    parser.add_argument('--local-rank', type=int, help='Rank of distributed training')
    args = parser.parse_args()

    # Load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), yaml.SafeLoader)
    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    data_cfg = dataset_cfg.get(cfg.dataset)
    for k, v in data_cfg.items():
        cfg[k] = v

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.freeze()

    # Set random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Setup logger
    logx.initialize(logdir=cfg.work_dir, hparams=cfg, tensorboard=True,
                    global_rank=args.local_rank if dist.is_initialized() else 0)
    logx.msg(pformat(cfg))
    shutil.copytree('models', os.path.join(cfg.work_dir, 'models'), dirs_exist_ok=True)

    # Setup CUDNN and GPU device
    torch.backends.cudnn.benchmark = True
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    start_time = time.monotonic()
    train(cfg)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
