# Code for Neural Reflectance Surfaces (NeRS)

[[`arXiv`](https://arxiv.org/abs/2110.07604)]
[[`Project Page`](https://jasonyzhang.com/ners/)]
[[`Colab Demo`](https://colab.research.google.com/drive/1L4Sl_9Osc2J_I5YpkteLrb-VbnwdDokd?usp=sharing)]
[[`Bibtex`](#CitingNeRS)]

This repo contains the code for NeRS: Neural Reflectance Surfaces.

The code was tested with the following dependencies:
* Python 3.8.6
* Pytorch 1.7.0
* Pytorch3d 0.5.0
* CUDA 11.0

## Installation

### Setup

We recommend using conda to manage dependencies. Make sure to install a cudatoolkit
compatible with your GPU.

```
git clone git@github.com:jasonyzhang/ners.git
conda create -n ners python=3.8
conda activate ners
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
pip install -r requirements.txt
```

### Installing Pytorch3d

Here, we list the recommended steps for installing Pytorch3d. Refer to the 
[official installation directions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
for troubleshooting and additional details.

```
mkdir -p external
git clone --depth 1 --branch v0.5.0 https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
cd external/pytorch3d
conda activate ners
conda install -c conda-forge -c fvcore -c iopath -c bottler fvcore iopath nvidiacub
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

We recommend beginning with the [demo notebook](notebooks/NeRS%20In-the-wild%20Demo.ipynb)
so that you can visualize the intermediate outputs. The demo notebook generates the 3D
reconstruction and illumination prediction for the espresso machine (data included). You
can also run the demo script:

```
python main.py --instance-dir data/espresso --symmetrize --export-mesh --predict-illumination
```

We also provide a [Colab notebook](https://colab.research.google.com/drive/1L4Sl_9Osc2J_I5YpkteLrb-VbnwdDokd?usp=sharing)
that runs on a single GPU. Note that the Colab demo does not include the view-dependent
illumination prediction. At the end of the demo, you can view the turntable NeRS
rendering and download the generated mesh as an obj.

To run on your own objects, you will need to acquire images and masks. See
`data/espresso` for an example of the expected directory structure.

We also provide the images and masks for all objects in the paper. All objects except
hydrant and robot should have a `--symmetrize` flag.
```
gdown https://drive.google.com/uc?id=1JWuofTIlcLJmmzYtZYM2SvZVizJCcOU_
unzip -n misc_objects.zip -d data
```


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