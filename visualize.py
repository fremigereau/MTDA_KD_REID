import torch as torch
import torchreid
from torchreid.utils import Logger
import sys
import os.path as osp
import torch.optim
import torch.nn as nn

batch_size = 32
max_epoch = 1
use_tensorboard = False

sources = ["msmt17"]
targets = ["market1501","dukemtmcreid"]
# targets = ["market1501"]

loss = 'feat_kd_margin_relu'
student_model_arch = 'resnet50'
teacher_model_arch = 'resnet50'

one_by_one = True

optim = 'sgd'
lr = 0.01

sched = 'single_step'
step_size = 5

# path_model_student = "log/source_training_early_stop/msmt17/resnet18/source_msmt17_best.pth.tar-60"
path_model_student = "log/source_training_early_stop/msmt17/resnet50_fc_layers/source_msmt17.pth.tar-80"

path_model_teacher_list = ["/export/livia/home/vision/dmekhazni/SpCL/examples/logs/spcl_uda_good_save/msmt2market_resnet50_torchreid/model_best.pth.tar",
                   "/export/livia/home/vision/dmekhazni/SpCL/examples/logs/spcl_uda_good_save/msmt2dukemtmc_resnet50_torchreid/model_best.pth.tar"]
# path_model_teacher_list = ["log/source_training_early_stop/market1501/resnet18/source_market1501.pth.tar-105",
#                            "log/source_training_early_stop/dukemtmcreid/resnet18/source_dukemtmcreid.pth.tar-85",
#                            "log/source_training_early_stop/cuhk03/resnet18/source_cuhk03.pth.tar-120"]


log_dir = 'log/test_gradients/no_feat_match_lkd_target/{sources}_to_{targets}_{teacher}_t_{student}_s_loss_{loss}/optim_{optim}_lr_{lr}_sched_{scheduler}_step_{step_size}_{epochs}_epochs_regular'.format(
                                                                        sources='_'.join([str(elem) for elem in sources]),
                                                                        targets='_'.join([str(elem) for elem in targets]),
                                                                        teacher=teacher_model_arch,
                                                                        student=student_model_arch,
                                                                        loss=loss,
                                                                        optim=optim,
                                                                        lr=lr,
                                                                        scheduler=sched,
                                                                        step_size=step_size,
                                                                        epochs=max_epoch,
                                                                        one_by_one="_one_by_one" if one_by_one else "")

tmp_dir = log_dir
counter = 1
while osp.exists(log_dir):
        log_dir = tmp_dir + "_" + str(counter)
        counter += 1

log_name = 'console_txt.log'
sys.stdout = Logger(osp.join(log_dir, log_name))
print("Saving experiment data to : {}".format(log_dir))

datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=sources,
        targets=targets,
        height=256,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True,
        workers=0
)

print("Initialize model student")
model_student, optimizer_student, scheduler_student, start_epoch = torchreid.initialize_model_optimizer_scheduler(
        name=student_model_arch, num_classes=datamanager.num_train_pids, loss=loss, pretrained=True,
        optimizer_type=optim, lr=lr,
        lr_scheduler=sched, stepsize=step_size,
        path_model=path_model_student,
        teacher_arch=teacher_model_arch,
        load_optim=False
        )

print("Initialize model(s) teacher")
models_teacher_list = list()
optimizer_teacher_list = list()
scheduler_teacher_list = list()

for i in range(0, len(datamanager.targets)):
        model, optimizer, scheduler, start_epoch = torchreid.initialize_model_optimizer_scheduler(
                # name=teacher_model_arch, num_classes=datamanager.list_train_loader_t[i].dataset.num_train_pids,
                name=teacher_model_arch, num_classes=0,
                loss=loss, pretrained=True,
                optimizer_type=optim, lr=lr,
                lr_scheduler=sched, stepsize=step_size,
                # path_model=path_model_teacher,
                path_model=path_model_teacher_list[i],
                # None means this is a teacher network
                teacher_arch=None,
                load_optim=False,
                spcl=True
        )
        models_teacher_list.append(model)
        optimizer_teacher_list.append(optimizer)
        scheduler_teacher_list.append(scheduler)



engine = torchreid.engine.Visual_Engine(
        datamanager=datamanager,
        model_student=model_student,
        optimizer_student=optimizer_student,
        scheduler_student=scheduler_student,
        models_teacher_list=models_teacher_list,
        optimizer_teacher_list=optimizer_teacher_list,
        scheduler_teacher_list=scheduler_teacher_list,
        label_smooth=True,
        mmd_only=False
)


# engine.run(
#         save_dir=log_dir,
#         test_only=True,
#         visrank=True,
#         eval_teachers=False
# )

start_epoch = 0
# Start the domain adaptation
engine.run(
        save_dir=log_dir,
        max_epoch=max_epoch,
        eval_freq=5,
        print_freq=50,
        test_only=False,
        visrank=False,
        start_epoch=start_epoch,
        use_tensorboard=use_tensorboard,
        eval_teachers=False
)

current_ep = 0
