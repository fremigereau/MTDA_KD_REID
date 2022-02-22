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


class AdvMTDAEngine(Engine):

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
            mmd_only=True,
            distiller=None,
            distiller_optim=None,
            weight_kd=0.005
    ):
        super(AdvMTDAEngine, self).__init__(
            datamanager,
            model_student, optimizer_student, scheduler_student,
            models_teacher_list, optimizer_teacher_list, scheduler_teacher_list,
            use_gpu, mmd_only)

        self.distiller = distiller
        self.distiller_optim = distiller_optim


        self.weight_kd = weight_kd
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
            global_only=False,
            distance_only=True,
            all=False
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
            open_layers=None,
    ):

        AverageMeter_List = []
        for d in range(0, len(self.models_teacher_list)):
            AverageMeter_List.append(StoreAverageMeters())


        self.Ne = max_epoch
        growth_rate = torch.log(torch.FloatTensor([self.final_value / self.starting_value])) / torch.FloatTensor([self.Ne])
        beta = self.starting_value * torch.exp(growth_rate * (epoch))
        beta = beta.cuda()
        # beta = beta.to(device=torch.device('cuda:0'))

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
        num_batches = 0
        for train_loader in self.datamanager.list_train_loader_t:
            num_batches += len(train_loader)
        end = time.time()

        for i, (model_teacher, optimizer_teacher, scheduler_teacher, train_loader_t) \
                in enumerate(zip(
                    self.models_teacher_list,
                    self.optimizer_teacher_list,
                    self.scheduler_teacher_list,
                    self.datamanager.list_train_loader_t))\
                :
            dataset_target = "_" + self.datamanager.targets[i]
            model_teacher.train()
            self.model_student.train()

            # ------------------------------------------------------------------------------------------------------- #
            for batch_idx, (data, data_t) in enumerate(zip(self.train_loader, train_loader_t)):
                AverageMeter_List[i].data_time.update(time.time() - end)

                imgs, pids = self._parse_data_for_train(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()
                    # imgs = imgs.to(device = torch.device('cuda:0'))
                    # pids = pids.to(device=torch.device('cuda:0'))

                imgs_t, pids_t = self._parse_data_for_train(data_t)
                if self.use_gpu:
                    imgs_t = imgs_t.cuda()
                    # imgs_t = imgs_t.to(device = torch.device('cuda:0'))

                self.optimizer_student.zero_grad()
                optimizer_teacher.zero_grad()

                # # 1) Compute L_DA and optimize Teacher
                # name = "L_DA"
                #
                # # outputs_Source_Teacher, features_Source_Teacher = model_teacher(imgs)    # Source
                # # _, features_Target_Teacher = model_teacher(imgs_t)  # Target
                #
                # _, pre_classifier_outputs_Source_Teacher, outputs_Source_Teacher = model_teacher(imgs)    # Source
                # _, pre_classifier_outputs_Target_Teacher, outputs_Target_Teacher = model_teacher(imgs_t)  # Target
                #
                # #   a) Compute L_DC
                #
                # loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(
                #     criterion=self.criterion_mmd,
                #     outputs=pre_classifier_outputs_Source_Teacher,
                #     targets=pre_classifier_outputs_Target_Teacher
                # )
                # l_dmmd_teacher = loss_mmd_wc + loss_mmd_bc + loss_mmd_global
                #
                # #   b) Compute L_CE
                # loss_t_teacher = self._compute_loss(criterion=self.criterion_t, outputs=pre_classifier_outputs_Source_Teacher, targets=pids)  # source triplet
                # loss_x_teacher = self._compute_loss(criterion=self.criterion_x, outputs=outputs_Source_Teacher, targets=pids)   # source softmax
                # l_ce_teacher = loss_t_teacher + loss_x_teacher
                #
                # #   c) Sum both losses
                #
                # l_da = l_ce_teacher + l_dmmd_teacher
                # l_da = (1-beta) * l_da
                #
                #
                # #   d) Compute gradient (loss.backward) and optimize model (optimizer.step())
                #
                # l_da.backward()
                # optimizer_teacher.step()
                #
                # self.optimizer_student.zero_grad()
                # optimizer_teacher.zero_grad()
                #
                # #   e) Printing and tensorboard update
                #
                # AverageMeter_List[i].batch_time.update(time.time() - end)
                # AverageMeter_List[i].losses_ce_teacher.update(l_ce_teacher.item(), pids.size(0))
                # AverageMeter_List[i].losses_dmmd_teacher.update(l_dmmd_teacher.item(), pids.size(0))
                # AverageMeter_List[i].losses_da.update(l_da.item(), pids.size(0))
                #
                num_batches_target = self.datamanager.list_train_loader_t[0].sampler.length / self.datamanager.train_loader.batch_size
                # if writer is not None:
                #     n_iter = epoch * num_batches_target + batch_idx
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_ce_teacher', AverageMeter_List[i].losses_ce_teacher.avg, n_iter)
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_dmmd_teacher', AverageMeter_List[i].losses_dmmd_teacher.avg, n_iter)
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_da', AverageMeter_List[i].losses_da.avg, n_iter)

                # -------------------------------------------------------------------------------------------------- #

                features_Source_Student, pre_class_features_Source_Student, outputs_Source_Student = self.model_student(imgs)
                features_Target_Student, pre_class_features_Target_Student, outputs_Target_Student = self.model_student(imgs_t)
                with torch.no_grad():
                    features_Source_Teacher, pre_class_features_Source_Teacher, outputs_Source_Teacher = model_teacher(imgs)
                    features_Target_Teacher, pre_class_features_Target_Teacher, outputs_Target_Teacher = model_teacher(imgs_t)

                # for (feat_s, feat_t) in zip(features_Source_Student, features_Target_Student):
                loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(
                    criterion=self.criterion_mmd,
                    outputs=pre_class_features_Source_Student,
                    targets=pre_class_features_Target_Student
                )
                l_d_mmd_student_source_target = loss_mmd_wc + loss_mmd_bc + loss_mmd_global

                loss_t_student = self._compute_loss(criterion=self.criterion_t, outputs=pre_class_features_Source_Student, targets=pids)  # source triplet
                loss_x_student = self._compute_loss(criterion=self.criterion_x, outputs=outputs_Source_Student, targets=pids)


                stu_s_kd_loss = 0.
                teacher_s_kd_loss = 0.

                self.distiller(features_Source_Teacher, features_Source_Student)

                for j in range(len(features_Source_Teacher)):
                    domain_s = torch.zeros(features_Source_Student[j].size(0)).long().cuda()
                    domain_t = torch.ones(features_Source_Student[j].size(0)).long().cuda()
                    stu_s_kd_loss += self._compute_loss(
                    criterion=self.criterion_focal,
                    outputs=features_Source_Student[j],
                    targets=domain_s
                    )
                    teacher_s_kd_loss += self._compute_loss(
                    criterion=self.criterion_focal,
                    outputs=features_Source_Teacher[j],
                    targets=domain_t
                    )
                l_kd_source = self.weight_kd * (stu_s_kd_loss + teacher_s_kd_loss)

                self.distiller(features_Target_Teacher,features_Target_Student)

                stu_t_kd_loss = 0.
                teacher_t_kd_loss = 0.

                for j in range(len(features_Target_Teacher)):
                    domain_s = torch.zeros(features_Target_Student[j].size(0)).long().cuda()
                    domain_t = torch.ones(features_Target_Student[j].size(0)).long().cuda()
                    stu_t_kd_loss += self._compute_loss(
                        criterion=self.criterion_focal,
                        outputs=features_Target_Student[j],
                        targets=domain_s
                    )
                    teacher_t_kd_loss += self._compute_loss(
                        criterion=self.criterion_focal,
                        outputs=features_Target_Teacher[j],
                        targets=domain_t
                    )
                l_kd_target = self.weight_kd * (stu_t_kd_loss + teacher_t_kd_loss)

                loss_kd = l_kd_source + l_kd_target + loss_t_student + loss_x_student + l_d_mmd_student_source_target
                # loss_kd = l_kd_source + loss_t_student + loss_x_student + l_d_mmd_student_source_target
                # loss_kd = l_kd_target + loss_t_student + loss_x_student + l_d_mmd_student_source_target
                # loss_kd = beta * loss_kd

                #       v) Printing and tensorboard update

                AverageMeter_List[i].batch_time.update(time.time() - end)
                AverageMeter_List[i].losses_dmmd_student.update(l_d_mmd_student_source_target.item(), pids.size(0))
                AverageMeter_List[i].losses_ce_student.update(loss_x_student.item(), pids.size(0))
                AverageMeter_List[i].losses_triplet_student.update(loss_t_student.item(), pids.size(0))
                AverageMeter_List[i].losses_kd_source.update(l_kd_source.item(), pids.size(0))
                AverageMeter_List[i].losses_kd_target.update(l_kd_target.item(), pids.size(0))
                AverageMeter_List[i].losses_kd.update(loss_kd.item(), pids.size(0))

                # if writer is not None:
                #     n_iter = epoch * num_batches_target + batch_idx
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_kd_source', AverageMeter_List[i].losses_kd_source.avg, n_iter)
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_kd_target', AverageMeter_List[i].losses_kd_target.avg, n_iter)
                #     writer.add_scalar('Train_' + name + dataset_target + '/losses_kd', AverageMeter_List[i].losses_kd.avg, n_iter)


                loss_kd.backward()
                self.optimizer_student.step()
                # self.scheduler_student.step()
                # self.distiller_optim.step()

                self.optimizer_student.zero_grad()
                # self.distiller_optim.zero_grad()
                optimizer_teacher.zero_grad()

    # -------------------------------------------------------------------------------------------------------------- #


                if writer is not None:
                    n_iter = epoch * num_batches_target + batch_idx
                    writer.add_scalar('Train_INFO_' + dataset_target + '/Time', AverageMeter_List[i].batch_time.avg, n_iter)
                    writer.add_scalar('Train_INFO_' + dataset_target + '/Beta', beta, n_iter)

                if (batch_idx + 1) % print_freq == 0:
                    # estimate remaining time
                    eta_seconds = AverageMeter_List[i].batch_time.avg * (
                            num_batches_target - (batch_idx + 1) + (max_epoch -
                                                             (epoch + 1)) * num_batches_target
                    )
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(
                        'Epoch: [{0}/{1}][{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'L_CE_s {losses1.val:.4f} ({losses1.avg:.4f})\t'
                        'L_DMMD_s {losses2.val:.4f} ({losses2.avg:.4f})\t'
                        'L_Tri_s {losses3.val:.4f} ({losses3.avg:.4f})\t'
                        'L_KD_Source {losses4.val:.4f} ({losses4.avg:.4f})\t'
                        'L_KD_Target {losses5.val:.4f} ({losses5.avg:.4f})\t'
                        'L_KD {losses6.val:.4f} ({losses6.avg:.4f})\t'
                        'l_r {lr}\t'
                        'eta {eta}'.format(
                            epoch + 1,
                            max_epoch,
                            batch_idx + 1,
                            num_batches,
                            batch_time=AverageMeter_List[i].batch_time,
                            losses1=AverageMeter_List[i].losses_ce_student,
                            losses2=AverageMeter_List[i].losses_dmmd_student,
                            losses3=AverageMeter_List[i].losses_triplet_student,
                            losses4=AverageMeter_List[i].losses_kd_source,
                            losses5=AverageMeter_List[i].losses_kd_target,
                            losses6=AverageMeter_List[i].losses_kd,
                            lr=self.optimizer_student.param_groups[0]['lr'],
                            eta=eta_str
                        )
                    )

                end = time.time()

        if self.scheduler_student is not None:
            self.scheduler_student.step()

        if scheduler_teacher is not None:
            scheduler_teacher.step()
