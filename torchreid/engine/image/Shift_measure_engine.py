from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import KLDivergenceLoss, \
    MaximumMeanDiscrepancy, TripletLoss, CrossEntropyLoss, MarginLoss, FocalLoss, LogEuclidLoss
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


class DomainShiftEngine(Engine):

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
            mmd_only=True
    ):
        super(DomainShiftEngine, self).__init__(
            datamanager,
            model_student, optimizer_student, scheduler_student,
            models_teacher_list, optimizer_teacher_list, scheduler_teacher_list,
            use_gpu, mmd_only)

        self.criterion_kl = KLDivergenceLoss()
        self.criterion_mmd = MaximumMeanDiscrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=False
        )

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

        end = time.time()
        num_batches_target = 0
        for aLoader in self.datamanager.list_train_loader_t:
            num_batches_target += len(aLoader)

        target_num = len(self.datamanager.targets)
        iteration_order = np.arange(target_num)
        for ix in iteration_order:
            # ------------------------------------------------------------------------------------------------------- #
            loss_mmd = 0
            l_kl = 0
            for batch_idx, (data, data_t) in enumerate(
                    zip(self.train_loader, self.datamanager.list_train_loader_t[ix])):
                avg_meter_obj.data_time.update(time.time() - end)

                imgs, pids = self._parse_data_for_train(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()

                imgs_t, pids_t = self._parse_data_for_train(data_t)
                if self.use_gpu:
                    imgs_t = imgs_t.cuda()

                self.optimizer_student.zero_grad()

                # -------------------------------------------------------------------------------------------------- #
                with torch.no_grad():
                    features_Source_Student, pre_class_features_Source_Student, outputs_Source_Student = self.model_student(
                        imgs, target=ix)
                    features_Target_Student, pre_class_features_Target_Student, outputs_Target_Student = self.model_student(
                        imgs_t, target=ix)

                l_wc, l_bc, l_mmd = self._compute_loss(
                    criterion=self.criterion_mmd,
                    outputs=pre_class_features_Source_Student,
                    targets=pre_class_features_Target_Student
                )
                loss_mmd += (l_mmd+l_wc+l_bc)

                l_kl += self._compute_loss(
                    criterion=self.criterion_kl,
                    outputs=pre_class_features_Source_Student,
                    targets=pre_class_features_Target_Student
                )



                # -------------------------------------------------------------------------------------------------------------- #

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
                        'eta {eta}'.format(
                            epoch + 1,
                            max_epoch,
                            batch_idx + 1,
                            len(self.datamanager.list_train_loader_t[ix]),
                            batch_time=avg_meter_obj.batch_time,
                            target=self.datamanager.targets[ix],
                            eta=eta_str
                        )
                    )

                end = time.time()
            print('Shift going from ' + str(self.datamanager.sources) + ' to ' + str(self.datamanager.targets[ix]) + 'is:')
            print('D-MMD = ' + str(loss_mmd/len(self.datamanager.list_train_loader_t[ix])))
            print('KL Div = ' + str(l_kl/len(self.datamanager.list_train_loader_t[ix])))

