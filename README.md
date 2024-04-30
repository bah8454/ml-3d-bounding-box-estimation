# Examination and Experimentation upon 3D Bounding Box Estimation
## Introduction
This is an experiment and examination of [this GitHub repository](https://github.com/skhadem/3D-BoundingBox) which uses PyTorch to implement [this paper](https://arxiv.org/abs/1612.00496).  

![example-image](https://cdn.discordapp.com/attachments/1226010239195676797/1234657712722415656/image.png?ex=66318819&is=66303699&hm=2e82b6d79ccdf2f43d391159e5b7ed9d8b422562f00845d5144e637f83e8158b&)

## Dependencies and Installation
> **_NOTE:_** This application can be run on any computer that has viable tools for compiling the [OpenCV](https://opencv.org/) library.  This means that the CS lab computers cannot run this application by default, unless the GCC compatibility with openCV is reconsidered, which is outside the scope of this experiment.  This application has been tested successfully on multiple Windows machines.

1. Install [Python 3.6.8](https://www.python.org/downloads/release/python-368/)
2. Install OpenCV with `pip install opencv-python`.  This will take several minutes, as OpenCV requires heavy C++ compilation to function correctly.  If prompted, install the correct version of the Microsoft C++ Redistributable for your machine and use-case.  Once complete, you should have OpenCV version 4.9.0.80.
3. Install Toch and TorchVision (with CUDA support) with `pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`.  CUDA support is required, which means your machine must have a GPU.  The application is functional when using both the 960M and the 2070, but other models with CUDA support should also work as intended.
4. Download the test data from the [Kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset.  Download the left color images, the training labels, and the camera calibration matrices, which should be roughly 13 GB in total.  Unzip these downloads into the `Kitti` directory.

## Usage
1. Download the YOLOv3 weights from the official [source](https://pjreddie.com/media/files/yolov3.weights) and add them to the `weights` directory.  
2. Run `python Train.py` to train the model, or download a pre-trained `.pkl` weights file.  The loss function for
the orientation is driven to -1, so negative loss values are expected.
3. Run `python Run.py` to display the results on testing images, with the optional `--show-yolo` parameter to display the 2D bounding boxes.  Pressing SPACE will move to the next image.  Alternatively, run `python Run_no_yolo.py` to quickly jump through the testing images and print an average error value, which is inversely proportional to the tenability of the model. 

## How it works
The PyTorch neural net predicts the orientation and dimensions of objects in a given scene (static image) and then uses another neural network (YOLOv3) to obtain a 2d bounding box in order to calculate the full 3D location and projected bounding box.  It is assumed that the objects in question have zero pitch and roll, which is not a surprising assumption for cars, bicycles, and pedestrians.

## Future Goals
Utilize the extra data provided by the Kitti dataset, including the time sequences of images, which could allow 3D bounding box calculations to predict future locations by regressing over the velocity of corresponding boxes.

## Credit
This is a modified fork of [this repository](https://github.com/skhadem/3D-BoundingBox) which is a fork of [this repository](https://github.com/fuenwang/3D-BoundingBox), both of which are based on [this paper](https://arxiv.org/abs/1612.00496) which uses 2D bounding from [this paper](https://arxiv.org/abs/1607.07155).
