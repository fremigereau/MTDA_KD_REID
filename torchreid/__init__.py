from __future__ import print_function, absolute_import

import torchreid.models.resnet
from torchreid import data, optim, utils, engine, losses, models, metrics

from torchreid.models.dsbn import convert_bn, convert_dsbn

def initialize_model_optimizer_scheduler(
        name, num_classes, loss, pretrained, optimizer_type, lr, lr_scheduler, stepsize,
        path_model=None, teacher_arch=None, spcl=False, load_optim=True, pcgrad=False, fc_dim=2048, multi_head=0):
    if spcl:
        num_classes = 0
    model = models.build_model(
        name=name,
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        teacher_arch=teacher_arch,
        fc_dim=fc_dim
    )

    optimizer = optim.build_optimizer(
        model,
        optim=optimizer_type,
        lr=lr
    )

    if pcgrad:
        optimizer = optim.PCGrad(optimizer)
        scheduler = None
    else:
        scheduler = optim.build_lr_scheduler(
            optimizer,
            lr_scheduler=lr_scheduler,
            stepsize=stepsize
        )
    start_epoch = 0
    if path_model != None:
        if spcl:
            convert_dsbn(model)
        if load_optim == False:
            # We use pretrained model to continue the training on the target domain
            start_epoch = utils.resume_from_checkpoint(path_model, model, optimizer=None)
        else:
            # We use pretrained model to continue the training on the target domain
            start_epoch = utils.resume_from_checkpoint(path_model, model, optimizer)
    if spcl:
        convert_bn(model)
    if fc_dim > 0 and multi_head != 0:
        torchreid.models.resnet.convert_2_multi_head(model ,multi_head=multi_head)
    model = model.cuda()
    return model, optimizer, scheduler, start_epoch