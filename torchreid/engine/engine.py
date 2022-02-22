from __future__ import division, print_function, absolute_import
import time
import os
import os.path as osp
import sys
import datetime
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from collections import deque
from torchreid import metrics
from torchreid.utils import (
    AverageMeter, re_ranking, save_checkpoint, visualize_ranked_results, Logger, load_checkpoint
)
from torchreid.losses import DeepSupervision
import torch
import os
torch.multiprocessing.set_sharing_strategy('file_system')


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
            self,
            datamanager,
            model_student=None,
            optimizer_student=None,
            scheduler_student=None,
            models_teacher_list=[],
            optimizer_teacher_list=[],
            scheduler_teacher_list=[],

            use_gpu=True,
            mmd_only=True,
    ):
        self.datamanager = datamanager
        self.model_student = model_student
        self.optimizer_student = optimizer_student
        self.scheduler_student = scheduler_student
        self.models_teacher_list = models_teacher_list
        self.optimizer_teacher_list = optimizer_teacher_list
        self.scheduler_teacher_list = scheduler_teacher_list
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.train_loader_t = self.datamanager.train_loader_t
        self.mmd_only = mmd_only
        self.target_names = list(self.test_loader.keys())

    def run(
            self,
            save_dir='log',
            max_epoch=0,
            start_epoch=0,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
            start_eval=0,
            eval_freq=-1,
            test_only=False,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            use_tensorboard=True,
            eval_teachers=True

    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            save_best_only (bool, optional): during training, save the best model on test set and last epoch.
                Default is True to save storage.
        """

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        if test_only:
            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        self.save_dir = save_dir
        if self.writer is None and use_tensorboard:
            self.writer = SummaryWriter(log_dir=save_dir)
        else:
            self.writer = None # Test with no tensorboard

        time_start = time.time()

        print('=> Start training')

        best_r1 = 0
        best_epoch = 0

        for epoch in range(start_epoch, max_epoch):

            self.train(
                epoch,
                max_epoch,
                self.writer,
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            if (epoch + 1) >= start_eval \
                    and eval_freq > 0 \
                    and (epoch + 1) % eval_freq == 0 \
                    and (epoch + 1) != max_epoch:

                print("Test :")
                is_best = False
                rank1, mAP = self.test(
                    epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    eval_teachers=eval_teachers
                )

                if rank1 > best_r1:
                    best_r1 = rank1
                    is_best = True
                    best_epoch = epoch

                self._save_checkpoint(  # Student
                    model=self.model_student, target_name="source_" + self.datamanager.sources[0],
                    epoch=epoch, rank1=rank1, mAP=mAP, save_dir=save_dir, is_best=is_best
                )

                if len(self.models_teacher_list) > 0 and eval_teachers:
                    for m, n in zip(self.models_teacher_list, self.target_names):
                        self._save_checkpoint(model=m, target_name="target_" + n, epoch=epoch, rank1=0,
                                              save_dir=save_dir)


        if max_epoch > 0:
            print('=> Final test with best model from epoch ' + str(best_epoch))
            checkpoint = load_checkpoint(osp.join(save_dir, 'model-best.pth.tar'))
            self.model_student.load_state_dict(checkpoint['state_dict'])
            rank1, mAP = self.test(
                epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            # Delete useless files
            if osp.exists(osp.join(save_dir, 'checkpoint.pth.tar')):
                os.remove(osp.join(save_dir, 'checkpoint.pth.tar'))
            else:
                print(osp.join(save_dir, 'checkpoint.pth.tar') + " does not exist")
            if osp.exists(osp.join(save_dir, 'gallery_features.pickle')):
                os.remove(osp.join(save_dir, 'gallery_features.pickle'))
            else:
                print(osp.join(save_dir, 'gallery_features.pickle') + " does not exist")
            if osp.exists(osp.join(save_dir, 'query_features.pickle')):
                os.remove(osp.join(save_dir, 'query_features.pickle'))
            else:
                print(osp.join(save_dir, 'query_features.pickle') + " does not exist")

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python

            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::

            This must be implemented in subclasses.
        """
        raise NotImplementedError

    def test(
            self,
            epoch,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            save_dir='',
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            eval_teachers=True
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """

        targets = list(self.test_loader.keys())
        average_rank1 = 0
        average_mAP = 0

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1, mAP = self._evaluate(
                epoch,
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                eval_teachers=eval_teachers
            )

            average_rank1 += rank1
            average_mAP += mAP

        average_rank1 /= len(targets)
        average_mAP /= len(targets)

        return average_rank1, average_mAP

    @torch.no_grad()
    def _evaluate(
            self,
            epoch,
            dataset_name='',
            query_loader=None,
            gallery_loader=None,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            save_dir='',
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            eval_teachers=True
    ):
        batch_time = AverageMeter()
        trgt_idx = self.target_names.index(dataset_name)

        def _feature_extraction(data_loader, model, target=None):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self._parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self._extract_features(imgs, model, target=target)
                batch_time.update(time.time() - end)
                features = features.data.cpu()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        # Evaluate one time using the Student model (main objective)
        print('Extracting features from query set using model_student...')
        qf, q_pids, q_camids = _feature_extraction(query_loader, self.model_student, trgt_idx)
        with open(save_dir + '/query_features.pickle', 'wb') as f:
            pickle.dump([qf, q_pids], f)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set using model_student...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader, self.model_student, trgt_idx)
        with open(save_dir + '/gallery_features.pickle', 'wb') as f:
            pickle.dump([gf, g_pids], f)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03,
            dataset_name=dataset_name
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.
                    return_query_and_gallery_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        cmc_student = cmc[0]
        mAP_student = mAP

        if len(self.models_teacher_list) > 0 and eval_teachers:
            # Evalute one time using the Teacher model (debug objective)
            print('Extracting features from query set using model_teacher...')
            qf, q_pids, q_camids = _feature_extraction(query_loader, self.models_teacher_list[trgt_idx])
            with open(save_dir + '/query_features.pickle', 'wb') as f:
                pickle.dump([qf, q_pids], f)
            print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

            print('Extracting features from gallery set using model_teacher...')
            gf, g_pids, g_camids = _feature_extraction(gallery_loader, self.models_teacher_list[trgt_idx])
            with open(save_dir + '/gallery_features.pickle', 'wb') as f:
                pickle.dump([gf, g_pids], f)
            print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

            print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

            if normalize_feature:
                print('Normalzing features with L2 norm ...')
                qf = F.normalize(qf, p=2, dim=1)
                gf = F.normalize(gf, p=2, dim=1)

            print(
                'Computing distance matrix with metric={} ...'.format(dist_metric)
            )
            distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
            distmat = distmat.numpy()

            if rerank:
                print('Applying person re-ranking ...')
                distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
                distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)

            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                use_metric_cuhk03=use_metric_cuhk03
            )

            print('** Results **')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

            if visrank:
                visualize_ranked_results(
                    distmat,
                    self.datamanager.
                        return_query_and_gallery_by_name(dataset_name),
                    self.datamanager.data_type,
                    width=self.datamanager.width,
                    height=self.datamanager.height,
                    save_dir=osp.join(save_dir, 'visrank_' + dataset_name + '_teacher'),
                    topk=visrank_topk
                )

        return cmc_student, mAP_student

    def _compute_loss(self, criterion, outputs, targets, deepsuper=False, source_only=False):
        if source_only:
            loss = criterion(outputs, targets)
        elif deepsuper:
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input, model, target=None):
        model.eval()  # put in eval mode
        return model(input,target=target)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, model, target_name, epoch, rank1, save_dir, mAP=0, is_best=False):
        save_checkpoint(
            model,
            {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                'mAP': mAP,
                'optimizer': self.optimizer_student.state_dict(),
                'scheduler': self.scheduler_student.state_dict() if (self.scheduler_student != None) else None,
            },
            save_dir,
            is_best=is_best,
            target_name=target_name
        )



"""
def evaluate_all(distmat, query=None, gallery=None,
                     query_ids=None, gallery_ids=None,
                     query_cams=None, gallery_cams=None, use_metric_cuhk03=False,
                     ranks=[1, 5, 10, 20]):

        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
        else:
            assert (query_ids is not None and gallery_ids is not None
                    and query_cams is not None and gallery_cams is not None)

        # Compute mean AP

        cmc, mAP = metrics.evaluate_rank(
            distmat,
            query_ids,
            gallery_ids,
            query_cams,
            gallery_cams,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results for Validation set **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        # Use the allshots cmc top-1 score for validation criterion
        return cmc"""
