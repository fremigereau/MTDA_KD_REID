from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import KLDivergenceLoss, \
    MaximumMeanDiscrepancy, TripletLoss, CrossEntropyLoss, MarginLoss, LogEuclidLoss
from ..engine import Engine
import torch
from itertools import cycle
import numpy as np

class StoreAverageMeters(object):
    def __init__(self):
        self.losses_dmmd_teacher = AverageMeter()
        self.losses_ce_teacher = AverageMeter()
        self.losses_da = AverageMeter()

        self.losses_ce_student = AverageMeter()
        self.losses_triplet_student = AverageMeter()
        self.losses_dmmd_student = AverageMeter()

        self.losses_kd_source = AverageMeter()
        self.losses_kd_target = AverageMeter()
        self.losses_kd = AverageMeter()


        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()


class MTDAEnginePerBatch(Engine):

    def __init__(
            self,
            datamanager,
            model_student,
            optimizer_student,
            scheduler_student=None,
            models_teacher_list=[],
            optimizer_teacher_list=[],
            scheduler_teacher_list=[],
            margin=0.3,
            use_gpu=True,
            label_smooth=True,
            mmd_only=True,
            kd_style='all',
            lda_weight=0,
            lkds_weight=0,
            lkdt_weight=1,
            target_order='random',
            log_loss=False
    ):
        super(MTDAEnginePerBatch, self).__init__(
            datamanager,
            model_student, optimizer_student, scheduler_student,
            models_teacher_list, optimizer_teacher_list, scheduler_teacher_list,
            use_gpu, mmd_only)

        self.starting_value = 0.1
        self.final_value = 0.8
        self.kd_style = kd_style
        self.lda_weight = lda_weight
        self.lkds_weight = lkds_weight
        self.lkdt_weight = lkdt_weight
        self.target_order = target_order
        self.log_loss = log_loss

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_kl = KLDivergenceLoss()
        
        self.criterion_mmd = MaximumMeanDiscrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=False
        )
        self.criterion_margin = MarginLoss()
        self.criterion_log_euclid = LogEuclidLoss(log=False)

    def train(
            self,
            epoch,
            max_epoch,
            writer,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None
    ):

        avg_meter_obj = StoreAverageMeters()

        self.model_student.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model_student, open_layers)
        else:
            open_all_layers(self.model_student)

        batch_size = self.datamanager.train_loader.batch_size
        end = time.time()
        num_batches_target = 0
        iter_t_list = []
        longest_loader = 0
        for aLoader in self.datamanager.list_train_loader_t:
            if (len(aLoader) > longest_loader):
                longest_loader = len(aLoader)
            tmp_iter = iter(aLoader)
            iter_t_list.append(tmp_iter)
            num_batches_target += len(aLoader)

        iter_s = iter(self.train_loader)

        self.model_student.train()
        for model_t in self.models_teacher_list:
            model_t.train()

            for batch_idx in range(1, longest_loader + 1):

                try:
                    data = iter_s.next()
                except StopIteration:
                    iter_s = iter(self.train_loader)
                    data = iter_s.next()
                imgs, pids = self._parse_data_for_train(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()

                # load all batches for all targets
                target_num = len(self.datamanager.targets)
                data_t = [None] * target_num
                imgs_t = [None] * target_num
                pids_t = [None] * target_num
                dist = [None] * target_num
                for t, it in enumerate(iter_t_list):
                    try:
                        data_t[t] = it.next()
                    except StopIteration:
                        it = iter(self.datamanager.list_train_loader_t[t])
                        data_t[t] = it.next()

                    imgs_t[t], pids_t[t] = self._parse_data_for_train(data_t[t])
                    if self.use_gpu:
                        imgs_t[t] = imgs_t[t].cuda()

                    if self.target_order == 'shift':
                        with torch.no_grad():
                            features_Source_Student, pre_class_features_Source_Student, outputs_Source_Student = self.model_student(
                                imgs,target=t)
                            features_Target_Student, pre_class_features_Target_Student, outputs_Target_Student = self.model_student(
                                imgs_t[t],target=t)
                        _, _, loss_mmd_global = self._compute_loss(
                            criterion=self.criterion_mmd,
                            outputs=pre_class_features_Source_Student,
                            targets=pre_class_features_Target_Student
                        )
                        dist[t] = loss_mmd_global.item()
                if self.target_order == 'shift':
                    iteration_order = np.argsort(dist)
                elif self.target_order == 'fixed':
                    target_num = len(self.datamanager.targets)
                    iteration_order = np.arange(target_num)
                elif self.target_order == 'random':
                    target_num = len(self.datamanager.targets)
                    iteration_order = np.arange(target_num)
                    np.random.shuffle(iteration_order)

                self.optimizer_student.zero_grad()

                for ix in iteration_order:

                    features_Source_Student, pre_class_features_Source_Student, outputs_Source_Student = self.model_student(
                        imgs,target=ix)
                    features_Target_Student, pre_class_features_Target_Student, outputs_Target_Student = self.model_student(
                        imgs_t[ix],target=ix)
                    with torch.no_grad():
                        features_Source_Teacher, pre_class_features_Source_Teacher, outputs_Source_Teacher = \
                        self.models_teacher_list[ix](imgs,target=ix)
                        features_Target_Teacher, pre_class_features_Target_Teacher, outputs_Target_Teacher = \
                        self.models_teacher_list[ix](imgs_t[ix],target=ix)

                    if self.lda_weight > 0:
                        loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(
                            criterion=self.criterion_mmd,
                            outputs=pre_class_features_Source_Student,
                            targets=pre_class_features_Target_Student
                        )
                        l_d_mmd_student_source_target = loss_mmd_wc + loss_mmd_bc + loss_mmd_global

                        loss_t_student = self._compute_loss(criterion=self.criterion_t,
                                                            outputs=pre_class_features_Source_Student,
                                                            targets=pids)  # source triplet
                        loss_x_student = self._compute_loss(criterion=self.criterion_x, outputs=outputs_Source_Student,
                                                            targets=pids)

                        l_da = self.lda_weight * (l_d_mmd_student_source_target + loss_x_student + loss_t_student)/100
                        avg_meter_obj.losses_da.update(l_da.item(), pids.size(0))
                    else:
                        l_da = 0

                    if self.lkds_weight > 0:
                        l_margin_source_feat = 0
                        l_euclid_s = 0
                        if self.kd_style != 'only_outputs':
                            feat_num = len(features_Source_Teacher)
                            for j, (feat_s, feat_t) in enumerate(zip(features_Source_Student, features_Source_Teacher)):
                                loss_margin_s = self._compute_loss(
                                    criterion=self.criterion_margin,
                                    outputs=feat_s,
                                    targets=feat_t
                                )
                                l_margin_source_feat += loss_margin_s / (2 ** (feat_num - j - 1))
                            # Divide to make it smaller
                            l_margin_source_feat /= (1000 * self.datamanager.train_loader.batch_size)
                        if self.kd_style != 'only_feats':
                            l_euclid_s = self._compute_loss(
                                criterion=self.criterion_log_euclid,
                                outputs=pre_class_features_Source_Student,
                                targets=pre_class_features_Source_Teacher
                            )
                        l_kd_source = self.lkds_weight * (l_euclid_s + l_margin_source_feat)
                        avg_meter_obj.losses_kd_source.update(l_kd_source.item(), pids.size(0))
                    else:
                        l_kd_source = 0

                    if self.lkdt_weight > 0:
                        l_margin_target_feat = 0
                        l_euclid = 0
                        if self.kd_style != 'only_outputs':
                            feat_num = len(features_Target_Teacher)
                            for j, (feat_s, feat_t) in enumerate(zip(features_Target_Student, features_Target_Teacher)):
                                loss_margin = self._compute_loss(
                                    criterion=self.criterion_margin,
                                    outputs=feat_s,
                                    targets=feat_t
                                )
                                l_margin_target_feat += loss_margin / (2 ** (feat_num - j - 1))
                            # Divide to make it smaller
                            l_margin_target_feat /= (1000 * self.datamanager.train_loader.batch_size)
                        if self.kd_style != 'only_feats':
                            l_euclid = self._compute_loss(
                                criterion=self.criterion_log_euclid,
                                outputs=pre_class_features_Target_Student,
                                targets=pre_class_features_Target_Teacher
                            )
                        l_kd_target = self.lkdt_weight * (l_euclid + l_margin_target_feat)
                        avg_meter_obj.losses_kd_target.update(l_kd_target.item(), pids.size(0))
                    else:
                        l_kd_target = 0

                    loss_kd = l_kd_source + l_kd_target + l_da

                    #       v) Printing and tensorboard update

                    avg_meter_obj.batch_time.update(time.time() - end)
                    avg_meter_obj.losses_kd.update(loss_kd.item(), pids.size(0))

                    loss_kd.backward()
                    self.optimizer_student.step()

                    self.optimizer_student.zero_grad()

                # -------------------------------------------------------------------------------------------------------------- #

                if writer is not None:
                    n_iter = epoch * num_batches_target + batch_idx
                    writer.add_scalar('Train_INFO_' + str(ix) + '/Time', avg_meter_obj.batch_time.avg, n_iter)

                if (batch_idx + 1) % print_freq == 0:
                    # estimate remaining time
                    eta_seconds = avg_meter_obj.batch_time.avg * (
                            num_batches_target - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches_target
                    )
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(
                        'Epoch: [{0}/{1}][{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Current Target: {target}\t'
                        'L_DA {losses3.val:.4f} ({losses3.avg:.4f})\t'
                        'L_KD_Source {losses4.val:.4f} ({losses4.avg:.4f})\t'
                        'L_KD_Target {losses5.val:.4f} ({losses5.avg:.4f})\t'
                        'L_KD {losses6.val:.4f} ({losses6.avg:.4f})\t'
                        'eta {eta}'.format(
                            epoch + 1,
                            max_epoch,
                            batch_idx + 1,
                            len(self.datamanager.list_train_loader_t[ix]),
                            batch_time=avg_meter_obj.batch_time,
                            target=self.datamanager.targets[ix],
                            losses3=avg_meter_obj.losses_da,
                            losses4=avg_meter_obj.losses_kd_source,
                            losses5=avg_meter_obj.losses_kd_target,
                            losses6=avg_meter_obj.losses_kd,
                            eta=eta_str
                        )
                    )

                end = time.time()

            if self.scheduler_student is not None:
                self.scheduler_student.step()