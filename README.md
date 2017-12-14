# Overview
This repository provides source code and pre-trained models for brain tumor segmentation with BraTS dataset. The method is detailed in [1], and it won the 2nd place of MICCAI 2017 BraTS Challenge. 

This implementation is based on NiftyNet and Tensorflow. While NiftyNet provides more automatic pipelines for dataloading, training, testing and evaluation, this naive implementation only makes use of NiftyNet for network definition, so that it is lightweight and extensible. A demo that makes more use of NiftyNet for brain tumor segmentation is proivde [here][nfitynet_demo].
[nfitynet_demo]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/tree/dev/demos/BRATS17

If you use any resources in this repository, please cite the following papers:

* [1] Guotai Wang, Wenqi Li, Sebastien Ourselin, Tom Vercauteren. "[Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks.][acnn_arxiv]" arXiv preprint arXiv:1710.04043 (2017).
[acnn_arxiv]: https://arxiv.org/abs/1709.00382

* [2] Eli Gibson*, Wenqi Li*, Carole Sudre, Lucas Fidon, Dzhoshkun I. Shakir, Guotai Wang, Zach Eaton-Rosen, Robert Gray, Tom Doel, Yipeng Hu, Tom Whyntie, Parashkev Nachev, Marc Modat, Dean C. Barratt, Sébastien Ourselin, M. Jorge Cardoso^, Tom Vercauteren^.
"[NiftyNet: a deep-learning platform for medical imaging.][niftynet_arxiv]" arXiv preprint arXiv: 1709.03485 (2017). 
[niftynet_arxiv]: https://arxiv.org/abs/1709.03485

# Requirements
1, a CUDA compatable GPU with memoery larger than 6GB is recommended for training. For testing only, a CUDA compatable GPU may not be required.
2, tensorflow. Install tensorflow following instructions from [https://www.tensorflow.org/install/][tensorflow_install].
[tensorflow_install]: https://www.tensorflow.org/install/