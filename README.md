# [Unsupervised Domain Adaptation in the dissimilarity space for Person Re-identification](https://arxiv.org/abs/2007.13890 "Unsupervised Domain Adaptation in the dissimilarity space for Person Re-identification")

## Installation


Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.

```
    git clone https://github.com/fremigereau/MTDA_KD_REID

    # create environment
    cd KD-REID
    conda create --name kd-reid python=3.7
    conda activate kd-reid

    # install dependencies
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
```

## To reproduce experiments :

### 0. Preparation of data
The code is based on the framework by: 
https://github.com/KaiyangZhou/deep-person-reid

**Please arrange the data as proposed here:
**
https://kaiyangzhou.github.io/deep-person-reid/datasets.html

### 1. Train source domain

To train a model based on source use the script source_training.py with the desired arguments. To to train a resnet50 on MSMT17 as source dataset for example:
```
    python scripts/source_training.py -ds msmt17 -a resnet50 -data-dir $DATA
```
Source dataset can be Market1501, DukeMTMCReID, CUHK03 and MSMT17 using arguments market1501, dukemtmcreid, msmt17 and cuhk03 respectively. $DATA refers to the folder where you keep the datasets.

The model will be saved in this repo and will be used to perform the adaptation.

### 2. Apply Domain Adaptation using D-MMD to produce the teacher models
For this step you may train a teacher model using the D-MMD[^1] approach developed by *Mekhazni, et al.*. To adapt the model with MSMT17 source and Market1501 target use:

```
    python scripts/DMMD.py -ds msmt17 -dt market1501 -a resnet50 -data-dir $DATA -mp $MODEL_PATH
```

Here $MODEL_PATH is the path to the model that was trained in part 1. Repeat for each target domain until you have all teacher models adapted. The models are saved to be used in the step 3. This step can be skipped if you want to use teacher that you have already trained with another approach.

[^1]: Mekhazni, Djebril, et al. "Unsupervised domain adaptation in the dissimilarity space for person re-identification." European Conference on Computer Vision. Springer, Cham, 2020.

### 3. Kowledge distillation to a common backbone.
Finally use Knowledge distillation with the teachers from step2 and a new student model. To train a student model of architecture osnet_x0_25 with source MSMT17 and tragets Market1501, DukeMTMCReID and CUHK03 using resnet50 teachers produced in step 2. : 

```
    python scripts/KD-REID.py -ds msmt17 -dt market1501 dukemtmcreid cuhk03 -as osnet_x0_25 -at resnet50 -data-dir $DATA -mps $MODEL_PATH_STUDENT -mpt $MODEL_PATH_TEACHER_MARKET $MODEL_PATH_TEACHER_DUKE $MODEL_PATH_TEACHER_CUHK
```

Where $MODEL_PATH_STUDENT is the path to a student model trained using the code from step 1. and $MODEL_PATH_TEACHER_X is the path to the teacher model trained in step 2. for target X
