import torchreid


def initialize_model_optimizer_scheduler(
        name, num_classes, loss, pretrained, optim, lr, lr_scheduler, stepsize, path_model=None):

        model = torchreid.models.build_model(
                name=name,
                num_classes=num_classes,
                loss=loss,
                pretrained=pretrained,
        )
        model = model.cuda()

        model_grl = torchreid.models.build_model(name='DANN_GRL', num_classes=0)
        model_grl.cuda()
        #model_student = nn.DataParallel(model_student).cuda() # Comment previous line and uncomment this line for multi-gpu use

        optimizer = torchreid.optim.build_optimizer(
                model,
                optim=optim,
                lr=lr
        )

        scheduler = torchreid.optim.build_lr_scheduler(
                optimizer,
                lr_scheduler=lr_scheduler,
                stepsize=stepsize
        )

        start_epoch = 0
        if path_model != None:
                # We use pretrained model to continue the training on the target domain
                start_epoch = torchreid.utils.resume_from_checkpoint(
                        path_model,
                        model,
                        optimizer
                )

                model_student = model.cuda()
                #model_student = nn.DataParallel(model_student).cuda()  # Comment previous line and uncomment this line for multi-gpu use

                return model, model_grl, optimizer, scheduler, start_epoch

        return model, model_grl, optimizer, scheduler, start_epoch


batch_size = 32

datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources="dukemtmcreid",
        targets=["cuhk03", "msmt17"],
        height=256,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True
)

model_student, model_grl_student, optimizer_student, scheduler_student, start_epoch = initialize_model_optimizer_scheduler(
        name='resnet50', num_classes=datamanager.num_train_pids, loss='mmd', pretrained=True,
        optim='adam', lr=0.0003,
        lr_scheduler='single_step', stepsize=50,
        path_model="log/source_training/dukemtmcreid/model/model.pth.tar-30"
        )

models_teacher_list = list()
models_grl_teacher_list = list()
optimizer_teacher_list = list()
scheduler_teacher_list = list()
for i in range(0, len(datamanager.targets)):
        model, model_grl, optimizer, scheduler, start_epoch = initialize_model_optimizer_scheduler(
                name='resnet50', num_classes=datamanager.num_train_pids, loss='mmd', pretrained=True,
                optim='adam', lr=0.0003,
                lr_scheduler='single_step', stepsize=50,
                path_model="log/source_training/dukemtmcreid/model/model.pth.tar-30"
        )
        models_teacher_list.append(model)
        models_grl_teacher_list.append(model_grl)
        optimizer_teacher_list.append(optimizer)
        scheduler_teacher_list.append(scheduler)

model_student.load_state_dict(models_teacher_list[0].state_dict())

engine = torchreid.engine.ImageKLKDEnginev2(
        datamanager=datamanager,
        model_student=model_student,
        model_grl_student=model_grl_student,
        optimizer_student=optimizer_student,
        scheduler_student=scheduler_student,
        models_teacher_list=models_teacher_list,
        models_grl_teacher_list=models_grl_teacher_list,
        optimizer_teacher_list=optimizer_teacher_list,
        scheduler_teacher_list=scheduler_teacher_list,
        label_smooth=True,
        mmd_only=False
)

#engine.run(
#        test_only=True
#)

# Start the domain adaptation
engine.run(
        save_dir='log/duke_to_msmt_cuhk',
        max_epoch=300,
        eval_freq=1,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=start_epoch
)

current_ep = 0
