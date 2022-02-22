from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import KLDivergenceLoss, \
    MaximumMeanDiscrepancy, TripletLoss, CrossEntropyLoss, MarginLoss, FocalLoss
from ..engine import Engine
import torch
from itertools import cycle
import numpy as np

from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp

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


class Visual_Engine(Engine):

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
            weight_t=1,
            weight_x=1,
            use_gpu=True,
            label_smooth=True,
            mmd_only=True
    ):
        super(Visual_Engine, self).__init__(
            datamanager,
            model_student, optimizer_student, scheduler_student,
            models_teacher_list, optimizer_teacher_list, scheduler_teacher_list,
            use_gpu, mmd_only)


        self.weight_t = weight_t
        self.weight_x = weight_x
        self.starting_value = 0.1
        self.final_value = 0.8

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
            global_only=True,
            distance_only=False
        )
        self.criterion_margin = MarginLoss()
        self.criterion_focal = FocalLoss(class_num=2)

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


        self.model_student.train()
        for model_t in self.models_teacher_list:
            model_t.train()

        target_num = len(self.datamanager.targets)
        iteration_order = np.arange(target_num)

        for ix in iteration_order:
            # ------------------------------------------------------------------------------------------------------- #
            for batch_idx, (data, data_t) in enumerate(zip(self.train_loader, self.datamanager.list_train_loader_t[ix])):

                imgs, pids = self._parse_data_for_train(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()


                imgs_t, pids_t = self._parse_data_for_train(data_t)
                if self.use_gpu:
                    imgs_t = imgs_t.cuda()

                # --------------------------------------------------------------------------------------------------
                # features_Source_Student, pre_class_features_Source_Student, outputs_Source_Student = self.model_student(imgs)
                features_Target_Student, pre_class_features_Target_Student, outputs_Target_Student = self.model_student(imgs_t)
                # features_Source_Teacher, pre_class_features_Source_Teacher, outputs_Source_Teacher = self.models_teacher_list[0](imgs)
                features_Target_Teacher, pre_class_features_Target_Teacher, outputs_Target_Teacher = self.models_teacher_list[ix](imgs_t)

                feat_num = len(features_Target_Teacher)
                l_margin_target = 0
                for j, (feat_s, feat_t) in enumerate(zip(features_Target_Student, features_Target_Teacher)):
                    loss_margin = self._compute_loss(
                        criterion=self.criterion_margin,
                        outputs=feat_s,
                        targets=feat_t
                    )
                    l_margin_target += loss_margin / (2 ** (feat_num - j - 1))
                # Divide to make it smaller
                l_margin_target /= (1000*self.datamanager.train_loader.batch_size)

                l_margin_target.backward()


                print("visualize")

                from torchreid.metrics import compute_distance_matrix

                dts = compute_distance_matrix(pre_class_features_Target_Student, pre_class_features_Target_Student)
                dts_print = torch.triu(dts, diagonal=1).flatten()
                dts_print = dts_print[dts_print.nonzero()].flatten()

                dtt = compute_distance_matrix(pre_class_features_Target_Teacher, pre_class_features_Target_Teacher)
                dtt_print = torch.triu(dtt, diagonal=1).flatten()
                dtt_print = dtt_print[dtt_print.nonzero()].flatten()

                l_mmd_pairwise = self.criterion_mmd(dtt, dts)




                # plt.figure()
                # sns.distplot(dtt_print.detach().cpu().numpy(), bins='auto', kde=False, label='Targets seen by Teacher')
                # sns.distplot(dts_print.detach().cpu().numpy(), bins='auto', kde=False, label='Targets seen by Student')
                # plt.xlabel('Euclidean distance')
                # plt.ylabel('Number of apparition')
                # plt.title('Distributions of pairwise scores on' + self.datamanager.targets[ix])
                # plt.legend()
                # save_path = osp.join(self.save_dir,"Target " + self.datamanager.targets[ix] + " After DA.png")
                # plt.savefig(save_path)
                # plt.close()
    # -------------------------------------------------------------------------------------------------------------- #


