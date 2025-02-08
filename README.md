# SwinJSCC & Massive MIMO 
There are 2 version of the system, one with MIMO and one without MIMO.

## Installation
```
conda create -n swinjscc -c conda-forge python=3.10
conda activate swinjscc
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm
```
## Dataset
The dataset structure should look like this:
```
Dataset
└── HR_Image_dataset
    ├── DIV2K_train_HR # Put your images here
    └── clic2021
        └── test # Put your images here
```

## Usage

All pretrained models are in [Google Drive](https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc?usp=sharing).

```
python main.py --trainset DIV2K --testset CLIC21 --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --model_size base
```

### Running the SwinJSCC_w/o_SAandRA model as Inference
The command lines to use the system in both version are the same:
```
python main.py --trainset DIV2K --testset kodak -- distortion-metric MSE --model SwinJSCC_w/o_SAandRA model --channel-type awgn --C 96 -- multiple-snr 10 --model_size base
```
This method can be apply on your own images.

## Related links
* BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
* Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
* DIV2K image dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
* Kodak image dataset: http://r0k.us/graphics/kodak/
* CLIC image dataset:  http://compression.cc
