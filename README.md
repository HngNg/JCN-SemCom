# JCN Cooperative Semantic Communication
This is a research project working on a cooperative communication system with 4 nodes: source node, relay node, eavesdropper node and destination node.

The relay node is to support the communication between source node and destination node. On the other hand, the eavesdropper node is to simulate eavesdropping attacks on source-destination and relay-destination channel.

Swin Transformer architecture is used to implement semantic encoder and decoder, with the intuition similar to the way auto-encoder works. To prevent eavesdroppers from sniffing data, aftificial noise is jammed into the channels to disrupt the reconstruction process at eavesdropper node.

There are 2 version of the system, one is single-input single-output and one is multiple-input multiple-output.

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
python main.py --trainset DIV2K  --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --model_size base
```

### Running the SwinJSCC_w/o_SAandRA model as Inference
The command lines to use the system in both version are the same:
```
python SFST.py --trainset DIV2K --testset CLIC21 -- distortion-metric MSE --model SwinJSCC_w/o_SAandRA model --channel-type awgn --C 96 -- multiple-snr 10 --model_size base
```
This system can reconstruct your own images.

## Related links
* Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
* DIV2K image dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
* Kodak image dataset: http://r0k.us/graphics/kodak/
* CLIC image dataset:  http://compression.cc
