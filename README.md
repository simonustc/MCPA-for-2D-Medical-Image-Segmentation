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
* For the Synapse datasets we used are provided by TransUnet's authors.[Get processed data in this link](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). Please go to "MCPA_Synapse/datasets/README.md" for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). Please prepare data in the data directory:
```
├── MCPA_Synapse
    ├──Synapse
    │   ├── test_vol_h5
    │   │   ├── case0001.npy.h5
    │   │   └── *.npy.h5
    │   └── train_npz
    │       ├── case0005_slice000.npz
    │       └── *.npz
    └──lists
        ├── all.lst
        ├── test_vol.txt
        └── train.txt
```

* For the ACDC datasets, please sign up in the [official ACDC website and download the dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc). Or [Get processed data in this link](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) to use the preprocessed data for we proposed. Please prepare data in the data directory:

```
├── MCPA_ACDC
    ├──ACDC
    │   ├── test_vol_h5
    │   │   ├── case0001.npy.h5
    │   │   └── *.npy.h5
    │   └── train_npz
    │       ├── case0005_slice000.npz
    │       └── *.npz
    └──lists
        ├── all.lst
        ├── test_vol.txt
        └── train.txt
```

* For the datasets of vessel segmentation, please sign up in the [official DRIVE website and download the dataset](https://drive.grand-challenge.org/), [official CHASE_DB1 website and download the dataset](https://blogs.kingston.ac.uk/retinal/chasedb1/), [official HRF website and download the dataset](https://www5.cs.fau.de/research/data/fundus-images/), and [official ROSE website and download the dataset]([https://www5.cs.fau.de/research/data/fundus-images/](https://imed.nimte.ac.cn/dataofrose.html). Please prepare data in the data directory:
```
├── MCPA_vessel
    ├── Dataset
    │   ├── DRIVE
    │   │   ├── training
    │   │   │   ├── images
    │   │   │   │   └── 21_training.tif
    │   │   │   └──1st_manual
    │   │   │       └── 21_manual1.gif
    │   │   └── test
    │   │       ├── images  
    │   │       │   └── 01_test.tif
    │   │       └── 1st_manual
    │   │           └── 01_manual1.gif
    │   ├── CHASEDB1
    │   │   ├── training
    │   │   │   ├── images
    │   │   │   │   └── Image_01L.jpg
    │   │   │   └── 1st_manual
    │   │   │       └── Image_01L_1stHO.png
    │   │   └── test
    │   │       ├── images  
    │   │       │   └── Image_11L.jpg
    │   │       └── 1st_manual
    │   │           └── Image_11L_1stHO.png
    │   └── HRF
    │       └── all
    │           ├── images
    │           │   └── 01_test.tif
    │           └── 1st_manual
    │               └── 01_manual1.gif      
    └── prepare_dataset
        └── data_path_list
            ├── DRIVE
            │   ├── train.txt
            │   └── test.txt
            ├── CHASEDB1
            │   ├── train.txt
            │   └── test.txt
            └── HRF
                ├── train.txt
                └── test.txt
```

## Training

### MCPA_Synapse

`python MCPA_Synapse/train.py --cfg ./configs/Synapse.yaml --root_path ./Synapse/` 


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









