import argparse
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
    parser.add_argument("dataset", type=str)

    args = parser.parse_args()
    basename = os.path.basename(args.model_path)
    prefix = os.path.splitext(basename)[0]

    # extract feature
    cmd = "python{} extract.py {} {} {} ".format(sys.version[0], args.gpu, args.model_path, args.dataset)
    subprocess.check_call(cmd.strip().split(" "))

    # evaluation
    dirname = os.path.dirname(args.model_path)
    q_mat_path = os.path.join(dirname, 'query-%s.mat' % prefix)
    g_mat_path = os.path.join(dirname, 'gallery-%s.mat' % prefix)

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

    data_root = dataset_cfg.get(args.dataset).data_root
    perm = sio.loadmat(os.path.join(data_root, 'exp', 'rand_perm_cam.mat'))
    perm = perm['rand_perm_cam']
    mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cam_ids, g_feats, g_ids, g_cam_ids, g_img_paths, perm)
    print('mAP = %f , r1 = %f , r5 = %f , r10 = %f , r20 = %f' % (mAP, r1, r5, r10, r20))
