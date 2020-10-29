import argparse
import logging
import os
import subprocess
import sys

import scipy.io as sio

from utils.eval_sysu import eval_sysu
from configs.default.dataset import dataset_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, default=None)

    args = parser.parse_args()
    dataset, fname = args.model_path.split("/")[1], args.model_path.split("/")[-1]

    if args.dataset is not None:
        dataset = args.dataset

    prefix = os.path.splitext(fname)[0]

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # extract feature
    cmd = "python{} extract.py {} {} ".format(sys.version[0], args.gpu, args.model_path)
    if args.dataset is not None:
        cmd += "--dataset {}".format(args.dataset)

    subprocess.check_call(cmd.strip().split(" "))

    # evaluation
    q_mat_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    g_mat_path = 'features/%s/gallery-%s.mat' % (dataset, prefix)

    assert os.path.exists(q_mat_path)
    assert os.path.exists(g_mat_path)

    mat = sio.loadmat(q_mat_path)
    q_feats = mat["feat"]
    q_ids = mat["ids"].squeeze()
    q_cam_ids = mat["cam_ids"].squeeze()

    mat = sio.loadmat(g_mat_path)
    g_feats = mat["feat"]
    g_ids = mat["ids"].squeeze()
    g_cam_ids = mat["cam_ids"].squeeze()
    g_img_paths = mat['img_path'].squeeze()

    perm = sio.loadmat(os.path.join(dataset_cfg.get(dataset).data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']
    eval_sysu(q_feats, q_ids, q_cam_ids, g_feats, g_ids, g_cam_ids, g_img_paths, perm)
