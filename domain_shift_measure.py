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

    log_dir = 'log/final_tests/domain_shift/{sources}'.format(
                sources=args.dataset_source
                )
    log_name = 'console_txt.log'
    sys.stdout = Logger(osp.join(log_dir, log_name))
    print("Saving experiment data to : {}".format(log_dir))
    print("==========\nArgs:{}\n==========".format(args))

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
            cuhk03_classic_split=True
    )

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
            use_fc=args.fc
            )

    engine = torchreid.engine.DomainShiftEngine(
                    datamanager=datamanager,
                    model_student=model_student,
                    optimizer_student=optimizer_student,
                    scheduler_student=scheduler_student,
                    label_smooth=True,
                    mmd_only=False
            )

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
    parser.add_argument('-fc', action='store_false', default=True)
    parser.add_argument('--multi-head', action='store_true', default=False)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default='single_step')
    parser.add_argument('--step-size', type=int, default=5)
    # training configs
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-freq', type=int, default=5)
    parser.add_argument('--tensorboard', action='store_true', default=False)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'reid-data'))
    parser.add_argument('-mps', '--model-path-student', type=str, metavar='PATH')

    main()