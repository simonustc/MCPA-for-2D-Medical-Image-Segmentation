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

### Download pre-trained Shunted Self-Attention model (SSA-samll)
[Get pre-trained model in this link] (https://drive.google.com/drive/folders/15iZKXFT7apjUSoN2WUMAbb0tvJgyh3YP): Put pretrained SSA-samll into folder "MCPA_Synapse/pretrained/" and "MCPA_ACDC/pretrained/"


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

### MCPA_ACDC:

`python MCPA_ACDC/train.py --cfg ./config/ACDC.yaml --root_path ./ACDC/`

### MCPA_vessel:

#### DRIVE
`python MCPA_vessel/config.py --dataset DRIVE --train_data_path_list ./prepare_dataset/data_path_list/DRIVE/train.txt --test_data_path_list ./prepare_dataset/data_path_list/DRIVE/test.txt`

`python MCPA_vessel/train.py`

#### CHASEDB1
`python MCPA_vessel/config.py --dataset CHASEDB1 --train_data_path_list ./prepare_dataset/data_path_list/CHASEDB1/train.txt --test_data_path_list ./prepare_dataset/data_path_list/CHASEDB1/test.txt`

`python MCPA_vessel/train.py`

#### HRF
`python MCPA_vessel/config.py --dataset HRF --train_data_path_list ./prepare_dataset/data_path_list/HRF/train.txt --test_data_path_list ./prepare_dataset/data_path_list/HRF/test.txt`

`python MCPA_vessel/train.py`


## Test

### MCPA_Synapse

`python MCPA_Synapse/test.py --model_path .../epoch_399.pth` 

### MCPA_ACDC

`python MCPA_ACDC/test.py --model_path .../epoch_399.pth` 

### MCPA_vessel:

#### DRIVE
`python MCPA_vessel/config.py --dataset DRIVE --train_data_path_list ./prepare_dataset/data_path_list/DRIVE/train.txt --test_data_path_list ./prepare_dataset/data_path_list/DRIVE/test.txt`

`python MCPA_vessel/train.py`

#### CHASEDB1
`python MCPA_vessel/config.py --dataset CHASEDB1 --train_data_path_list ./prepare_dataset/data_path_list/CHASEDB1/train.txt --test_data_path_list ./prepare_dataset/data_path_list/CHASEDB1/test.txt`

`python MCPA_vessel/train.py`

#### HRF
`python MCPA_vessel/config.py --dataset HRF --train_data_path_list ./prepare_dataset/data_path_list/HRF/train.txt --test_data_path_list ./prepare_dataset/data_path_list/HRF/test.txt`

`python MCPA_vessel/train.py`


## Results and Models

### MCPA_Synapse

[Links to models](https://drive.google.com/drive/folders/1bAtCNYCFPNREPlEqOd-99goWUr9P_eKt)

### MCPA_ACDC

[Links to models]()

### DRIVE

[Links to models](https://drive.google.com/drive/folders/1-FqxL2V8rOpURrllmBOSMnliSJVsL7HN)

### CHASEDB1

[Links to models](https://drive.google.com/drive/folders/1CzBNRn_OZBdtd7cSn8f8vIDjl7WAFeH5)

### HRF

[Links to models](https://drive.google.com/drive/folders/1Gqa-CzupxTfZOoYrZI2YF1JIEHavp5Md)

### ROSE










