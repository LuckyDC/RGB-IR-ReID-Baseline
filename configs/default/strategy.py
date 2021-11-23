from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.work_dir = "baseline"

# setting for loader
strategy_cfg.sample_method = "random"
strategy_cfg.batch_size = 128
strategy_cfg.p_size = 16
strategy_cfg.k_size = 8

# settings for optimizer
strategy_cfg.optimizer = "sgd"
strategy_cfg.lr = 0.1
strategy_cfg.wd = 5e-4
strategy_cfg.momentum = 0.9
strategy_cfg.betas = (0.5, 0.999)
strategy_cfg.lr_step = (40,)

strategy_cfg.fp16 = False

strategy_cfg.num_epoch = 60

# settings for dataset
strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (256, 128)

# settings for augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = False
strategy_cfg.padding = 10

# settings for base architecture
strategy_cfg.last_stride = 1

# logging
strategy_cfg.eval_interval = -1
strategy_cfg.log_period = 10
