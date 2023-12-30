# EGU-Net-Pytorch

This repository contains the pytorch implementation of the EGU-Net neural network.

## What is EGU-Net?
- EGU-Net stands for Endmember-Guided Unmixing Network,
and it is a learning-based hyperspectral unmixing method.

- For more information, please refer to the original paper at the following URL.
https://ieeexplore.ieee.org/document/9444141

## What is this project for?
- The original source code for EGU-Net is available at https://github.com/danfenghong/IEEE_TNNLS_EGU-Net
- However, it is written in very outdated version of tensorflow
- It supports only python 3.7 & Tensorflow 1.x, making it somewhat cumbersome to use in modern systems
- This project attempts to re-implement EGU-Net using the most recent versions of python & pytorch!!!

## That's great, but how do I get started?
- First, install the required python dependencies using the following code.
```
pip install -r requirements.txt
```
- The above code should install all python dependencies required for running this project.
- Next, download the dataset using the provided shell script dataloader.sh.
```
./dataloader.sh
```
- The shell script will create a new folder called "Data" in the current directory, and download all the needed data files there.
- After setting the dataset up, you're now ready to go :)