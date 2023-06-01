# MCPA-for-2D-Medical-Image-Segmentation

## Installation

### Requirements

* argparse>=1.4.0
* numpy>=1.19.2
* tqdm>=4.62.1
* h5py>=3.2.1
* imgaug>=0.4.0
* scipy>=1.9.3
* timm>=0.4.12
* einops>=0.4.1
* opencv-python>=4.6.0.66
* torchvision>=0.7.0
* simpleitk>=2.0.2
* medpy>=0.4.0
* yaml>=0.2.5
* pyyaml>=5.4.1
* yacs>=0.1.8
* tensorboardx>=2.2


### Dataset Preparation
* For the Synapse datasets we used are provided by TransUnet's authors.[Get processed data in this link](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). Please go to "MCPA_Synapse/datasets/README.md" for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

* For the ACDC datasets, Please Sign up in the official ACDC website and download the dataset.[Get processed data in this link](https://www.creatis.insa-lyon.fr/Challenge/acdc).Or use the preprocessed data for research purposes.[Get processed data in this link](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd).



For ImageNet-LT and iNaturalist2018, please prepare data in the data directory:
```
datasets
├── data_txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_val.txt
    ├── ImageNet_LT_train.txt
    └── ImageNet_LT_test.txt

```

getting the txt files from [data_txt file Link](https://drive.google.com/drive/folders/1ssoFLGNB_TM-j4VNYtgx9lxfqvACz-8V?usp=sharing)

For CRI+PPL, change the `data_path` in `config/.../.yaml`;

For CRI+PPW+RIDE, change the `data_loader:{data_dir} in `./config/...json`.


## Training

one GPU for Imbalance cifar10 & cifar100, two GPUs for ImageNet-LT, and eight GPUs iNaturalist2018.

Backbone network can be resnet32 for Imbalance cifar10 & cifar100, resnet10 for ImageNet-LT, and resnet50 for iNaturalist2018.

### CRI+PPL

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10/cifar10_CRI.yaml`

`python train.py --cfg ./config/cifar100/cifar100_CRI.yaml`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet/imagenet_CRI.yaml`

#### ina2018:

`python train.py --cfg ./config/ina2018/ina2018_CRI.yaml`

### CRI+PPW+RIDE

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10.json`

`python train.py --cfg ./config/cifar100.json`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet.json`

#### ina2018:

`python train.py --cfg ./config/ina2018.json`


## Validation

### CRI+PPL

`python eval.py --cfg ./config/....yaml resume /path/ckps/...pth.tar`

### CRI+PPW+RIDE

`python eval.py --cfg ./config/....json --resume /path/...pth`


## Results and Models

### CRI+PPL

[Links to models](https://drive.google.com/drive/folders/1b932TjGm_-GcuN9Mq24aExk2uZK64LWy?usp=sharing)

### CRI+PPW+RIDE

[Links to models](https://drive.google.com/drive/folders/1Dqh0Jcs-lqKv0BkEJmMX8JJwnhCL7mhx?usp=sharing)









