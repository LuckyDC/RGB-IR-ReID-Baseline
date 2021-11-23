import os
import numpy as np
import torch
import scipy.io as sio
from runx.logx import logx
from ignite.engine import Events
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_sysu import eval_sysu
from configs.default.dataset import dataset_cfg


def get_trainer(model, optimizer, lr_scheduler=None, enable_amp=False, log_period=10, save_interval=10,
                query_loader=None, gallery_loader=None, eval_interval=None):
    # Trainer
    trainer = create_train_engine(model, optimizer, enable_amp)

    # Evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if eval_interval > 0 and query_loader and gallery_loader:
        evaluator = create_eval_engine(model)

    # Metric
    timer = Timer(average=True)
    kv_metric = AutoKVMetric()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch
        logx.msg('Epoch[{}] completed.'.format(epoch))

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % save_interval == 0:
            state_dict = model.state_dict()
            save_path = os.path.join(logx.logdir, 'checkpoint_ep{}.pt'.format(epoch))
            torch.save(state_dict, save_path)
            logx.msg("Model saved at {}".format(save_path))

        if evaluator and epoch % eval_interval == 0:
            torch.cuda.empty_cache()

            # extract query feature
            model.eval_mode = 'infrared'
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()

            # extract gallery feature
            model.eval_mode = 'visible'
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            perm_path = os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat')
            perm = sio.loadmat(perm_path)['rand_perm_cam']
            mAP, r1, r5, r10, r20 = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm)
            logx.msg('mAP = %f , r1 = %f , r5 = %f , r10 = %f , r20 = %f' % (mAP, r1, r5, r10, r20))

            val_metrics = {'mAP': mAP, 'rank-1': r1, 'rank-5': r5, 'rank-10': r10, 'rank-20': r20}
            logx.metric('val', val_metrics, epoch)

            # clear temporary storage
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
            for k in sorted(metric_dict.keys()):
                msg += "\t%s: %.4f" % (k, metric_dict[k])
            logx.msg(msg)
            logx.metric('train', metric_dict, iteration)

            kv_metric.reset()
            timer.reset()

    return trainer
