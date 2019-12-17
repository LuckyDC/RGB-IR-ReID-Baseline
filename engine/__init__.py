import logging
import os
import numpy as np
import torch
import scipy.io as sio

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_sysu import eval_sysu
from configs.default.dataset import dataset_cfg


def get_trainer(model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)

    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=3, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator = create_eval_engine(model, non_blocking)

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0:
            torch.cuda.empty_cache()

            # extract query feature
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            mAP, r1, r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm)

            if writer is not None:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()

        kv_metric.update(engine.state.output)

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iter_in_epoch % log_period == 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)

            metric_dict = kv_metric.compute()

            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)

                logger.info(msg)

            kv_metric.reset()
            timer.reset()

    return trainer
