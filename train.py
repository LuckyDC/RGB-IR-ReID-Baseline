import logging
import os
import pprint

import torch
import yaml
from apex import amp
from torch import optim

from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.baseline import Baseline


def train(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "log.txt",
                        filemode="a")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))

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
    model = Baseline(num_classes=cfg.num_id,
                     dual_path=cfg.dual_path,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     classification=cfg.classification)

    model.cuda()

    # optimizer
    assert cfg.optimizer in ['adam', 'sgd']
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)

    # convert model for mixed precision training
    model, optimizer = amp.initialize(model, optimizer, enabled=cfg.fp16, opt_level="O2")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=cfg.lr_step,
                                                  gamma=0.1)

    # engine
    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)
    engine = get_trainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)


if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/market2duke.yml")
    args = parser.parse_args()

    # set random seed
    seed = 0
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # enable cudnn backend
    torch.backends.cudnn.benchmark = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()

    train(cfg)
