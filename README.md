# Code for Neural Reflectance Surfaces (NeRS)

[[`arXiv`](https://arxiv.org/abs/2110.07604)]
[[`Project Page`](https://jasonyzhang.com/ners/)]
[[`Bibtex`](#CitingNeRS)]

This repo contains the code for NeRS: Neural Reflectance Surfaces.

The code was tested with the following dependencies:
* Python 3.8.6
* Pytorch 1.7.0
* Pytorch3d 0.4.0
* CUDA 11.0

## Installation

### Setup

We recommend using conda to manage dependencies. Make sure to install a cudatoolkit
compatible with your GPU.

```
git clone git@github.com:jasonyzhang/ners.git
conda create -n ners python=3.8
cond activate pytorch3d
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
pip install -r requirements.txt
```

### Installing Pytorch3d

Here, we list the recommended steps for installing Pytorch3d. Refer to the 
[official installation directions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
for troubleshooting and additional details.

```
mkdir -p external
git clone https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
cd external/pytorch3d
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install -c bottler nvidiacub
python setup.py install
```

If you need to compile for multiple architectures (e.g. Turing for 2080TI and Maxwell
for 1080TI), you can pass the architectures as an environment variable, i.e. 
`TORCH_CUDA_ARCH_LIST="Maxwell;Pascal;Turing;Volta" python setup.py install`.

If you get a warning about the default C/C++ compiler on your machine, you should
compile Pytorch3D using the same compiler that your pytorch installation uses, likely
gcc/g++. Try: `CC=gcc CXX=g++ python setup.py install`.

### Acquiring Object Masks

To get object masks, we recommend using 
[PointRend](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend)
for COCO classes or [GrabCut](https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html)
for other categories.

If using GrabCut, you can try [this interactive segmentation tool](https://github.com/jasonyzhang/interactive_grabcut).

## Running the Code

### Running on MVMC

Coming Soon!

### Running on Your Own Objects

We recommend beginning with the demo notebook so that you can visualize the intermediate
outputs. The demo notebook generates the 3D reconstruction and illumination prediction
for the espresso machine (data included).

To run on your own objects, you will need to acquire images and masks. See
`data/espresso` for an example of the expected directory structure.


## <a name="CitingNeRS"></a>Citing NeRS

If you use find this code helpful, please consider citing:

```BibTeX
@inproceedings{zhang2021ners,
  title={{NeRS}: Neural Reflectance Surfaces for Sparse-view 3D Reconstruction in the Wild},
  author={Zhang, Jason Y. and Yang, Gengshan and Tulsiani, Shubham and Ramanan, Deva},
  booktitle={Conference on Neural Information Processing Systems},
  year={2021}
}
```