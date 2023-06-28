## Installation

### Enviroment
- Ubuntu20.04 
- CUDA 11.1 (Not Essential)
- CUDNN 8.0.5 (Not Essential)
- NVIDIA RTX3080 (Not Essential)

### Requirements:

- PyTorch >= 1.8.0 Installation instructions can be found in https://pytorch.org/get-started/locally/.
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9,< 6.0
- OpenCV
- scikit-image
- nuscenes-devkit

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
conda create --name SAF-FCOS
conda activate SAF-FCOS

# this installs the right pip and dependencies for the fresh python
conda install ipython

# FCOS and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.2
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
export INSTALL_DIR=$PWD

# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
pip install pycocotools

# install nuScenes api.
pip install nuscenes-devkit

# install scikit-image
pip install scikit-image

# install opencv
pip install opencv-python

# install SAF-FCOS
cd $INSTALL_DIR
git clone https://github.com/Singingkettle/SAF-FCOS.git
cd SAF-FCOS

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop --no-deps


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```