import argparse
import os
import numpy as np
import scipy.io as sio
import torch

from configs.default import dataset_cfg
from data import get_test_loader
from models.baseline import Baseline
from models.mixnet import MixNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)  # TODO compatible for different models
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--dataset", type=str, default=None)

    args = parser.parse_args()
    model_path = args.model_path
    fname = model_path.split("/")[-1]

    if args.dataset is not None:
        dataset = args.dataset
    else:
        dataset = model_path.split("/")[1]

    prefix = os.path.splitext(fname)[0]

    dataset_config = dataset_cfg.get(dataset)
    image_size = (args.img_h, 128)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Baseline(eval=True, drop_last_stride=True, dual_path=False)
    # model = MixNet(eval=True, drop_last_stride=True)

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()
    model.cuda()

    # extract test feature
    gallery_loader, query_loader = get_test_loader(root=dataset_config.data_root,
                                                   batch_size=512,
                                                   image_size=image_size,
                                                   num_workers=16)
    # extract query features
    feats = []
    labels = []
    cam_ids = []
    img_paths = []
    for data, label, cam_id, img_path, _ in query_loader:
        with torch.autograd.no_grad():
            feat = model(data.cuda(non_blocking=True), cam_ids=cam_id)

        feats.append(feat.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        cam_ids.append(cam_id.data.cpu().numpy())
        img_paths.extend(img_path)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    cam_ids = np.concatenate(cam_ids, axis=0)
    print(feats.shape)

    dir_name = "features/{}".format(dataset, prefix)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    save_name = "{}/query-{}.mat".format(dir_name, prefix)
    sio.savemat(save_name,
                {"feat": feats,
                 "ids": labels,
                 "cam_ids": cam_ids,
                 "img_path": img_paths})

    # extract gallery features
    feats = []
    labels = []
    cam_ids = []
    img_paths = []
    for data, label, cam_id, img_path, _ in gallery_loader:
        with torch.autograd.no_grad():
            feat = model(data.cuda(non_blocking=True), cam_ids=cam_id)

        feats.append(feat.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        cam_ids.append(cam_id.data.cpu().numpy())
        img_paths.extend(img_path)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    cam_ids = np.concatenate(cam_ids, axis=0)
    print(feats.shape)

    dir_name = "features/{}".format(dataset, prefix)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    save_name = "{}/gallery-{}.mat".format(dir_name, prefix)
    sio.savemat(save_name,
                {"feat": feats,
                 "ids": labels,
                 "cam_ids": cam_ids,
                 "img_path": img_paths})
