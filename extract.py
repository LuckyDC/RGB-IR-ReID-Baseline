import argparse
import os
import numpy as np
import scipy.io as sio
import torch

from configs.default import dataset_cfg
from data import get_test_loader
from models.baseline import Baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)  # TODO compatible for different models
    parser.add_argument("dataset", type=str, default=None)
    parser.add_argument("--img-h", type=int, default=256)

    args = parser.parse_args()
    model_path = args.model_path
    basename = os.path.basename(model_path)
    prefix = os.path.splitext(basename)[0]

    dataset = args.dataset
    dataset_config = dataset_cfg.get(dataset)
    image_size = (args.img_h, 128)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Baseline()
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
    model.eval_mode = 'infrared'
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

    dirname = os.path.dirname(args.model_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    save_name = "{}/query-{}.mat".format(dirname, prefix)
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
    model.eval_mode = 'visible'
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

    save_name = "{}/gallery-{}.mat".format(dirname, prefix)
    sio.savemat(save_name,
                {"feat": feats,
                 "ids": labels,
                 "cam_ids": cam_ids,
                 "img_path": img_paths})
