# R2Plus1D-PyTorch
PyTorch implementation of the R2Plus1D convolution based ResNet architecture described in the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"

Link to original: [paper](https://arxiv.org/abs/1711.11248) and [code](https://github.com/facebookresearch/R2Plus1D)

***NOTE: This repository has been archived, although forks and other work that extend on top of this remain welcome***

## Requirements 

R2Plus1D-PyTorch has the following requirements

* PyTorch 0.4 and dependencies
* OpenCV (tested on 3.4.0.12)
* tqdm (for progress bars)

### About this repository

This repository consists of four python files:

* `module.py` - Contains an implementation of the factored, R2Plus1D convolution the entire implementation is based around. It is designed to be a replacement for nn.Conv3D in the appropriate scenario
* `network.py` - Uses `module.py` to build up the residual network described in the paper
* `dataset.py` - Implements a PyTorch dataset, that can load videos with appropriate labels from a given directory.
* `trainer.py` - A mildly modified version of the script from the PyTorch [tutorials](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) to train the model. Features saving and restoring capabilities. 

### Training on Kinetics-400/600

This repository does not include a crawler or downloader for the Kinetics-400/600 dataset, however, one can be found [here](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). It is strongly recommended to downsample the videos prior to training (and not on the fly), using a tool such as ffmpeg. If using the crawler, this can be done by adding `"-vf", "scale=172:128"` to the ffmpeg command list in the download clip function.

### Training in general

This repository is designed for the ResNet to be trained on any dataset of videos in general, using the VideoDataloader class from dataset.py . It expects the videos to be arranged in a directory -> [train/val] folders -> [class_label] folders (one for each class) -> videos (the files themselves). 

Forks and fixes of this repo are highly welcome!
