# 4D-StOP

Official code for 4D-StOP ([ECCV 2022 AVVision Workshop](https://avvision.xyz/eccv22/))!

**4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation**

[Lars Kreuzberg](), [Idil Esen Zulfikar](https://www.vision.rwth-aachen.de/person/245/), [Sabarinath Mahadevan](https://www.vision.rwth-aachen.de/person/218/), [Francis Engelmann](https://francisengelmann.github.io/) and [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/)

ECCV 2022 AVVision Workshop | [Paper] () | [Project Page] ()

![Teaser Image](images/overview.pdf)

<!--Official code for the paper "4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation".-->

<!--This repository contains the official code for the following paper:-->

<!--**4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation**  
Lars Kreuzberg, Idil Esen Zulfikar, Sabarinath Mahadevan, Francis Engelmann and Bastian Leibe  
Accepted to Advanced Autonomous Driving Workshop at ECCV 2022
arxiv-->

<!--In this work, we present a new paradigm called 4D-StOP to tackle the task of 4D Panoptic LiDAR Segmentation. 4D-StOP first generates spatio-temporal proposals using voting based center predictions, where each point in the 4D volume votes for a corresponding center. These tracklet proposals are further aggregated using learnt geometric features. The tracklet aggregation method effectively generates a video level 4D scene representation over the entire space-time volume. This is in contrast to the existing end-to-end trainable state-of-the-art approach which uses spatio-temporal embeddings that are represented by Gaussian probability functions. Our voting-based tracklet generation method followed by the geometric feature based aggregation technique generates a much better panoptic LiDAR segmentation quality as compared to modelling the entire 4D volume using Gaussian probabilities. 4D-StOP achieves a new state-of-the-art when applied to the SemanticKITTI test dataset with a score of 63.9 LSTQ, which is a large (+7%) improvement when compared to the current best performing end-to-end trainable method.-->

<!--4D-StOP is accepted at the ECCV 2022 [AVVision Workshop](https://avvision.xyz/eccv22/).-->

<!--This repository contains the official code for the paper "4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation", which is accepted to the Advanced Autonomous Driving Workshop at ECCV 2022.-->

## Installation
```
conda create --name <env> --file requirements.txt

cd cpp_wrappers
sh compile_wrappers.sh

cd pointnet2
python setup.py install
```
## Data
Download the SemanticKITTI dataset with labels from [here](http://semantic-kitti.org/dataset.html#download/).  
Add the semantic-kitti.yaml file to the folder.  
Create additional labels using `utils/create_center_label.py`.

Folder structure:
```
SemanticKitti/  
└── semantic-kitti.yaml  
└── sequences/  
    └── 00/  
        └── calib.txt  
        └── poses.txt  
        └── times.txt  
        └── labels  
            ├── 000000.label  
            ├── 000000.center.npy  
            ...  
         └── velodyne  
            ├── 000000.bin  
            ...
```
<!--
## Models
We uploaded two trained models in checkpoints/ that you can use for testing:   
Log_2022-06-13_17-33-24 -> 2-scan-setup  
Log_2022-06-17_12-16-59 -> 4-scan setup
-->

## Training
Use `train_SemanticKitti.py` for training. Adapt the config parameters like you wish. Importantly, set the paths for the dataset-folder, checkpoints-folders etc. In the experiments in our paper, we first train the model for 800 epochs setting `config.pre_train = True`. Then we train for further 300 epochs with `config.pre_train = False` and `config.freeze = True`. We train our models on a single NVIDIA A40 (48GB) GPU.

## Testing, Tracking and Evaluating
We provide an example script in `jobscript_test.sh`. You need to adapt the paths here. It executes `test_models.py` to generate the semantic and instance predictions within a 4D volume. In `test_models.py` you need to set config parameters and choose the model you want to test. To track instances across 4D volumes, `stitch_tracklets.py` is executed. To get the evaluation results `utils/evaluate_4dpanoptic.py` is used. We test our models on a single NVIDIA TitanX (12GB) GPU.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{kreuzberg2022stop,
  title={4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation},
  author={Kreuzberg, Lars and Zulfikar, Idil Esen and Mahadevan,Sabarinath and Engelmann, Francis and Leibe, Bastian},
  booktitle={European Conference on Computer Vision Workshop},
  year={2022}
}
```

## Acknowledgments
The code is based on the Pytoch implementation of [4D-PLS](https://github.com/MehmetAygun/4D-PLS), [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) and [VoteNet](https://github.com/facebookresearch/votenet).
