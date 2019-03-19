# 3Screen-CNN
A Python 3 and Keras library for training convolutional neural networks for image classification and semantic segmentation.

## Prequesites
* Python 3.5 or 3.6
* Tensorflow
* Keras
* NumPy
* Matplotlib
* Pillow
* scikit-learn
* scikit-image
* imgaug
* pandas
* NiBabel (only for brain tumor segmentation script)

### Installing

Clone or download this repository to your desired development directory and use pip to install the depedencies.

```
pip install imgaug==0.2.6
pip install Keras==2.2.4
pip install numpy==1.15.4
pip install pandas==0.23.4
pip install Pillow==5.3.0
pip install scikit-image==0.14.1
pip install scikit-learn==0.20.1
pip install nibabel==2.3.3
```

To install TensorFlow:
```
pip install tensorflow==1.12.0
```
or with GPU support
```
pip install tensorflow-gpu==1.12.0
```

There are several other steps to installing GPU-enabled Tensorflow. Follow the rest of the steps on https://www.tensorflow.org/install/ for GPU support.

### Project Organization
``` 
cnn/ 
```
The main module that contains all of the classes and functions for training a new CNN.
``` 
logs/ 
```
The default folder where all of the training logs and output csv files will save to during each training session.
``` 
sample_scripts/ 
```
Several scripts are given as examples on how to train a new CNN using the cnn module.
``` 
sample_data/ 
```
Several publically available biomedical datasets that are used in conjuction with the sample scripts. 
