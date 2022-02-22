import torchreid
from torchreid.utils import Logger, resume_from_checkpoint
import sys
import os.path as osp
import argparse


def main():
        args = parser.parse_args()


        log_dir = 'log/final_tests/source_training/{sources}{cuhk_split}/{model}'.format(sources=args.dataset_source,
                                                                             model=args.arch,
                                                                             cuhk_split='_new_cuhk' if args.new_cuhk else '')

        print("Saving experiment data to : {}".format(log_dir))
        log_name = 'console_txt.log'
        sys.stdout = Logger(osp.join(log_dir, log_name))

        print("==========\nArgs:{}\n==========".format(args))

        if args.new_cuhk:
            cuhk_classic = False
        else:
            cuhk_classic = True

        datamanager = torchreid.data.ImageDataManager(
                root='reid-data',
                sources=args.dataset_source,
                targets=args.dataset_source,
                height=args.height,
                width=args.width,
                batch_size_train=args.batch_size,
                batch_size_test=100,
                transforms=['random_flip', 'random_crop', 'random_erasing'],
                num_instances=args.num_instances,
                train_sampler='RandomIdentitySampler',
                load_train_targets=False,
                workers=args.workers,
                cuhk03_labeled=True,
                cuhk03_classic_split=cuhk_classic
        )

        model = torchreid.models.build_model(
                name=args.arch,
                num_classes=datamanager.num_train_pids,
                loss='triplet',
                pretrained=True,
                fc_dim=args.features
        )

        if args.arch == "mobilenetv2":
            start_epoch = resume_from_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/mobilenet_pretrain/mobilenetv2_1.4-bc1cc36b.pth' , model, optimizer=None)
        elif args.arch == "osnet_x1_0":
            start_epoch = resume_from_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/osnet_pretrain/osnet_x1_0_imagenet.pth', model, optimizer=None, ignore_classifier=True)
        elif args.arch == "osnet_x0_75":
            start_epoch = resume_from_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/osnet_pretrain/osnet_x0_75_imagenet.pth', model, optimizer=None, ignore_classifier=True)
        elif args.arch == "osnet_x0_5":
            start_epoch = resume_from_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/osnet_pretrain/osnet_x0_5_imagenet.pth',model, optimizer=None, ignore_classifier=True)
        elif args.arch == "osnet_x0_25":
            start_epoch = resume_from_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/osnet_pretrain/osnet_x0_25_imagenet.pth',model, optimizer=None, ignore_classifier=True)

        optimizer = torchreid.optim.build_optimizer(
                model,
                optim=args.optimizer,
                lr=args.lr
        )

        scheduler = torchreid.optim.build_lr_scheduler(
                optimizer,
                lr_scheduler=args.scheduler,
                stepsize=args.step_size
        )

        model = model.cuda()
        # model = nn.DataParallel(model).cuda()  # Comment previous line and uncomment this line for multi-gpu use

        start_epoch = 0

        engine = torchreid.engine.ImageTripletEngine(
                datamanager=datamanager,
                model_student=model,
                optimizer_student=optimizer,
                scheduler_student=scheduler,
                label_smooth=True,
        )

        if args.dataset_source == 'msmt17':
                fixbase_epoch = 10
                open_layers=['classifier']
        else:
                fixbase_epoch = 0
                open_layers = None

        engine.run(
                save_dir=log_dir,
                max_epoch=args.epochs,
                eval_freq=args.eval_freq,
                print_freq=args.print_freq,
                test_only=False,
                visrank=False,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
                start_epoch=start_epoch,
                use_tensorboard=False
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on source for D-MMD")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, nargs='+', default='msmt17')
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scheduler', type=str, default='single_step')
    parser.add_argument('--step-size', type=int, default=50)
    # training configs
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))

    main()
