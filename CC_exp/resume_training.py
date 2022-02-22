import torchreid
import os
import sys
import glob
import pathlib


"""path = pathlib.Path().absolute()
path = str(path) + "/log"

list_of_files = glob.glob(path) # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print latest_file

target = "market_to_all3"""""

simu = "market_to_all3"
epoch = "62"
sources = "market1501"
targets = ["dukemtmcreid", "msmt17", "cuhk03"]

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
        sources=sources,
        targets=targets,
        height=256,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True
)


path_model_source = "log/" + simu + "/source_" + datamanager.sources[0] + ".pth.tar-" + epoch

path_target = "log/" + simu + "/target_"
path_models_target = []
for i in datamanager.targets:
        path_models_target.append(path_target + i + ".pth.tar-" + epoch)


model_student, model_grl_student, optimizer_student, scheduler_student, start_epoch = initialize_model_optimizer_scheduler(
        name='resnet50', num_classes=datamanager.num_train_pids, loss='mmd', pretrained=True,
        optim='adam', lr=0.0003,
        lr_scheduler='single_step', stepsize=50,
        path_model=path_model_source
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
                path_model=path_models_target[i]
        )

        models_teacher_list.append(model)
        models_grl_teacher_list.append(model_grl)
        optimizer_teacher_list.append(optimizer)
        scheduler_teacher_list.append(scheduler)


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
        save_dir='log/' + simu,
        max_epoch=300,
        eval_freq=1,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=start_epoch
)
