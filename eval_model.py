import sys
import os.path as osp
import torch.nn as nn
from torchreid import models, optim, engine, utils, data
from torchreid.utils import model_complexity
from torchreid.models.dsbn import convert_bn, convert_dsbn


source = 'msmt17'
target = ['market1501','dukemtmcreid','cuhk03']

arch = 'osnet_x0_25'
model_path = '/export/livia/home/vision/fremigereau/workspace/SPCL_fremig/examples/logs/spcl_uda/msmt2market+duke+cuhk_osnet25_torchreid/model_best.pth.tar'

img_height = 256
img_width = 128

log_dir = 'log/eval_complex/testing'

log_name = 'console_txt.log'
sys.stdout = utils.Logger(osp.join(log_dir, log_name))
print("Saving experiment data to : {}".format(log_dir))

datamanager = data.ImageDataManager(
        root='reid-data',
        sources=source,
        targets=target,
        height=img_height,
        width=img_width,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True,
        blend=False,
        cuhk03_labeled=True,
        cuhk03_classic_split=True
)

print("Initialize model 1")
model = models.build_model(
        name=arch,
        # num_classes=datamanager.num_train_pids,
        num_classes=0,
        loss='kd_reid',
        pretrained=True,
        teacher_arch=None,
        fc_dim=2048)

convert_dsbn(model)

print("Initialize model 2")
model_2 = models.build_model(
        name=arch,
        num_classes=datamanager.num_train_pids,
        loss='kd_reid',
        pretrained=True,
        teacher_arch=None,
        fc_dim=0)

# check1 = utils.load_checkpoint(model_path)
check2 = utils.load_checkpoint('/export/livia/home/vision/fremigereau/workspace/KD-Reid/log/final_tests/KD_ReID_LKD_target_D-MMD/msmt172market1501+dukemtmcreid+cuhk03/resnet50_t_resnet18_s/_target_alt_batch_kd_style_only_outputs/model-best.pth.tar')
#
utils.resume_from_checkpoint(model_path,model)

convert_bn(model)

model = model.cuda()

engine = engine.ImageMmdEngine(
        datamanager=datamanager,
        model=model,
        optimizer=None,
        scheduler=None,
        label_smooth=True,
        mmd_only=False
)

print('Model 1 complexity')
num_params, flops = model_complexity.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)

print('Model 2 complexity')
num_params_2, flops_2 = model_complexity.compute_model_complexity(model_2, (1, 3, 256, 128), verbose=True)

engine.run(
        save_dir=log_dir,
        test_only=True,
        use_metric_cuhk03=True
)
