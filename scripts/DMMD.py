import torchreid
from torchreid.utils import Logger
import sys
import os.path as osp
import argparse
import torch.nn as nn

def main():
    args = parser.parse_args()

    blend = True if len(args.dataset_target) > 1 else False


    log_dir = 'log/final_tests/DMMD_ST/{sources}{cuhk_split}/{sources}_to_{targets}_{student}_s'.format(
                                                                            sources= args.dataset_source if isinstance(args.dataset_source, str) else
                                                                            '+'.join([str(elem) for elem in args.dataset_source]),
                                                                            targets=args.dataset_target if isinstance(args.dataset_target, str) else
                                                                            '+'.join([str(elem) for elem in args.dataset_target]),
                                                                            student=args.arch,
                                                                            cuhk_split='_new_cuhk' if args.new_cuhk else ''
                                                                            )

    log_name = 'console_txt.log'
    sys.stdout = Logger(osp.join(log_dir, log_name))
    print("Saving experiment data to : {}".format(log_dir))
    print("==========\nArgs:{}\n==========".format(args))

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
            blend=blend,
            cuhk03_labeled=True,
            cuhk03_classic_split=cuhk_classic
    )

    print("Initialize model student")
    model_student, optimizer_student, scheduler_student, start_epoch = torchreid.initialize_model_optimizer_scheduler(
            name=args.arch, num_classes=datamanager.num_train_pids, loss='mmd', pretrained=True,
            optimizer_type=args.optimizer, lr=args.lr,
            lr_scheduler=args.scheduler, stepsize=args.step_size,
            path_model=args.model_path,
            teacher_arch=None,
            load_optim = False,
            fc_dim=args.features
            )

    engine = torchreid.engine.ImageMmdEngine(
            datamanager=datamanager,
            model=model_student,
            optimizer=optimizer_student,
            scheduler=scheduler_student,
            label_smooth=True,
            mmd_only=False
    )

    # engine.run(
    #         save_dir=log_dir,
    #         test_only=True,
    #         use_metric_cuhk03=True
    # )
    start_epoch=0
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
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--scheduler', type=str, default='single_step')
    parser.add_argument('--step-size', type=int, default=50)
    # training configs
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    # pathmp
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('-mp', '--model-path', type=str, metavar='PATH')

    main()