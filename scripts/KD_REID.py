import torch as torch
import torchreid
from torchreid.utils import Logger
import sys
import os.path as osp
import torch.optim
import torch.nn as nn
import argparse

def main():
    args = parser.parse_args()

    log_dir = 'log/KD_ReID_{stda}/{sources}2{targets}{cuhk_split}/{teacher}_t_{student}_s/{target_alt}{kd_style}{target_order}{lkd_s_w}{lda_w}'.format(
                stda=args.stda,
                target_alt='' if args.target_alt == 'full' else '_target_alt_' + args.target_alt,
                kd_style='_kd_style_outs_feats' if args.kd_style == 'all' else '_kd_style_' + args.kd_style,
                target_order='' if args.target_order == 'random' else '_' + args.target_order + '_order',
                sources=args.dataset_source if isinstance(args.dataset_source, str) else
                '+'.join([str(elem) for elem in args.dataset_source]),
                targets=args.dataset_target if isinstance(args.dataset_target, str) else
                '+'.join([str(elem) for elem in args.dataset_target]),
                teacher=args.arch_teacher,
                student=args.arch_student,
                lkd_s_w='' if args.lkds_weight == 0 else '_lkds_' + str(args.lkds_weight),
                lda_w='' if args.lda_weight == 0 else '_lda_' + str(args.lda_weight),
                cuhk_split='_new_cuhk' if args.new_cuhk else ''
                )
    log_name = 'console_txt.log'
    sys.stdout = Logger(osp.join(log_dir, log_name))
    print("Saving experiment data to : {}".format(log_dir))
    print("==========\nArgs:{}\n==========".format(args))

    if args.multi_head:
        multi_head = len(args.dataset_target)
    else:
        multi_head = 0

    if args.new_cuhk:
        cuhk_classic = False
    else:
        cuhk_classic = True

    datamanager = torchreid.data.ImageDataManager(
            root=args.data_dir,
            sources=args.dataset_source,
            targets=args.dataset_target,
            height=args.height,
            width=args.width,
            batch_size_train=args.batch_size,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop', 'random_erasing'],
            num_instances=args.num_instances,
            train_sampler='RandomIdentitySampler',
            load_train_targets=True,
            workers=args.workers,
            cuhk03_labeled=True,
            cuhk03_classic_split=cuhk_classic
    )
    if args.stda == 'D-MMD':
        num_classes = datamanager.num_train_pids
        spcl = False
    else:
        num_classes = 0
        spcl = True

    print("Initialize model student")
    model_student, optimizer_student, scheduler_student, start_epoch = torchreid.initialize_model_optimizer_scheduler(
            name=args.arch_student, num_classes=datamanager.num_train_pids,
            loss='kd_reid', pretrained=True,
            optimizer_type=args.optimizer, lr=args.lr,
            lr_scheduler=args.scheduler, stepsize=args.step_size,
            path_model=args.model_path_student,
            teacher_arch=args.arch_teacher,
            spcl=False,
            load_optim=False,
            pcgrad=False,
            fc_dim=args.features,
            multi_head=multi_head
            )

    print("Initialize model(s) teacher")
    models_teacher_list = list()
    optimizer_teacher_list = list()
    scheduler_teacher_list = list()

    for i in range(0, len(datamanager.targets)):
            model, optimizer, scheduler, start_epoch = torchreid.initialize_model_optimizer_scheduler(
                    name=args.arch_teacher, num_classes=num_classes,
                    loss='kd_reid', pretrained=True,
                    optimizer_type=args.optimizer, lr=args.lr,
                    lr_scheduler=args.scheduler, stepsize=args.step_size,
                    path_model=args.model_path_teachers[i],
                    teacher_arch=None,
                    spcl=spcl,
                    load_optim=False,
                    fc_dim=args.features
            )
            models_teacher_list.append(model)
            optimizer_teacher_list.append(optimizer)
            scheduler_teacher_list.append(scheduler)

    if args.target_alt == 'full':
            engine = torchreid.engine.KDMTDAEngineOnebyOne(
                    datamanager=datamanager,
                    model_student=model_student,
                    optimizer_student=optimizer_student,
                    scheduler_student=scheduler_student,
                    models_teacher_list=models_teacher_list,
                    optimizer_teacher_list=optimizer_teacher_list,
                    scheduler_teacher_list=scheduler_teacher_list,
                    label_smooth=True,
                    mmd_only=False,
                    kd_style=args.kd_style,
                    lda_weight=args.lda_weight,
                    lkds_weight=args.lkds_weight,
                    lkdt_weight=args.lkdt_weight,
                    target_order=args.target_order,
                    log_loss=args.log_loss
            )
    elif args.target_alt == 'batch':
            engine = torchreid.engine.MTDAEnginePerBatch(
                datamanager=datamanager,
                model_student=model_student,
                optimizer_student=optimizer_student,
                scheduler_student=scheduler_student,
                models_teacher_list=models_teacher_list,
                optimizer_teacher_list=optimizer_teacher_list,
                scheduler_teacher_list=scheduler_teacher_list,
                label_smooth=True,
                mmd_only=False,
                kd_style=args.kd_style,
                lda_weight=args.lda_weight,
                lkds_weight=args.lkds_weight,
                lkdt_weight=args.lkdt_weight,
                target_order=args.target_order,
                log_loss=args.log_loss
            )
    else:
        engine = None

    # engine.run(
    #         save_dir=log_dir,
    #         test_only=True
    # )

    start_epoch = 0
    # Start the domain adaptation
    engine.run(
            save_dir=log_dir,
            max_epoch=args.epochs,
            eval_freq=args.eval_freq,
            print_freq=args.print_freq,
            test_only=False,
            visrank=False,
            start_epoch=start_epoch,
            use_tensorboard=args.tensorboard,
            eval_teachers=False,
            use_metric_cuhk03=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on source for D-MMD")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='msmt17')
    parser.add_argument('-dt', '--dataset-target',type=str, nargs='+', default='market1501')
    parser.add_argument('--new-cuhk', action='store_true', default=False)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-at', '--arch-teacher', type=str, default='resnet50')
    parser.add_argument('-as', '--arch-student', type=str, default='resnet50')
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--multi-head', action='store_true', default=False)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default='single_step')
    parser.add_argument('--step-size', type=int, default=5)
    # training configs
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-freq', type=int, default=5)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--target-alt', type=str ,choices=['full','batch','combined'], default='full')
    parser.add_argument('--kd-style', type=str ,choices=['all', 'only_feats', 'only_outputs'], default='all')
    parser.add_argument('--lda_weight', type=float, default=0)
    parser.add_argument('--lkds_weight', type=float, default=0)
    parser.add_argument('--lkdt_weight', type=float, default=1)
    parser.add_argument('--target-order', type=str ,choices=['random', 'fixed', 'shift'], default='random')
    parser.add_argument('--stda', type=str ,choices=['D-MMD', 'SPCL'], default='D-MMD')
    parser.add_argument('--log-loss', action='store_true', default=False)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'reid-data'))
    parser.add_argument('-mps', '--model-path-student', type=str, metavar='PATH')
    parser.add_argument('-mpt', '--model-path-teachers',type=str, nargs='+', metavar='PATH')

    main()