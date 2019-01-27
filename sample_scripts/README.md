## Creating a script for a custom data set
There are three simple steps to using this module to train a CNN on a given data set. 

1. Create a configuration class that inheirits from the parent class ```Config``` and override the attributes.
```python
from am_cnn.config import Config

class ExampleConfig(Config):
  NUM_EPOCHS = 5
  MODEL_NAME = 'inceptionv3'

```

2. Create a dataset class that inheirits from the parent class ```Dataset``` and override the ```load_data()``` method.
```python
import os
import PIL.Image as pil
from am_cnn.utils import Dataset

class ExampleDataset(Dataset):
  def load_data(path, input_shape):
    files = next(os.walk(path))[2]
    self.X["all] = np.zeros(((len(files), ) + input_shape), np.float32)
    self.y["all] = np.zeros(len(files), np.float32)
    for idx, file in enumerate(files):
      self.X["all][idx] = np.array(pil.open(os.path.join(path, file)))
      self.y["all"][idx] = 1 if 'class1' in file else 0
```

3. Create a main function that creates a ```Dataset```, ```Config```, and ```Model``` object calls the ```train()``` method of the ```Model``` object.

```python
from am_cnn.model import Model

if __name__ == "__main__":
  # Create the Config object
  config = ExampleConfig()
  
  # Create the Dataset object and load the data
  dataset = ExampleDataset()
  dataset.load_data()
  
  # Split the data into training, testing and validation
  dataset.split_data(test_size=config.TEST_SPLIT, validation_size=config.VAL_SPLIT)
  
  # Create the Model object and train
  model = Model()
  model.train(dataset, config)
  
```
