from yacs.config import CfgNode

dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "/home/chuanchen_luo/data/SYSU"
