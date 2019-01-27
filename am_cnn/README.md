
## config.py
This file contains the configuration parent class **Config** which is used to enter parameters used for a new training session. When creating a new training script, create new Config child class and override each of the default attributes of the parent class. A full description of each attribute can be found inside of the config.py file.

## utils.py
The **Dataset** parent class is defined in this file which holds two important attributes ```X``` and ```y```, which are dicts with the fields: ```all```, ```"full"```, ```"train"```, ```"test"```, ```"validation"```. This class contains all of the relavant data for training and evaluation and has several important methods for manipulating or augmenting the data. 

When creating a new training script, create a new ```Dataset``` child class and override the ```load_data()``` method since every dataset is storred slightly different. Use the load_data() method to load data into the "train", "test" and "validation" fields of the X and y attributes, or load all of the data into the ```"all"``` field of the ```X``` and ```y``` attributes and use the ```split_data()``` to split the data into the desired ratios.

## model.py
This contains the **Model** class which is responsible for training and evaulating a CNN on a given ```Dataset``` and ```Config``` object. There are several important methods in this class including ```train()```, ```evaluate()``` and several other useful methods for visualizing predicted classes or segmentation maps. The ```model``` attribute is the actual Keras model.

## networks.py
Several popular CNN architectures built using Keras. 

**Classification**
* Vgg16
* InceptionV3

**Segmentation**
* U-net and variations
