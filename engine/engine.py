import torch
import numpy as np
from torch.cuda import amp
from ignite.engine import Engine
from ignite.engine import Events


def create_train_engine(model, optimizer, enable_amp=False):
    device = torch.device("cuda", torch.cuda.current_device())
    scaler = amp.GradScaler(enabled=enable_amp)

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        cam_ids = cam_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=enable_amp):
            loss, metric = model(data, labels=labels, cam_ids=cam_ids, epoch=epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return metric

    return Engine(_process_func)


def create_eval_engine(model):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=False))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # cam list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
