## Creating a script for a custom data set
There are three simple steps to using this module to train a CNN on a given data set. 

1. Create a configuration class that inheirits from the parent class ```Config``` and override the attributes.
```python
from visikol_cnn.config import Config

class ExampleConfig(Config):
  DATA_PATH = '../Example Dataset'
  NUM_EPOCHS = 5
  MODEL_NAME = 'inceptionv3'

```

2. Create a dataset class that inheirits from the parent class ```Dataset``` and override the ```load_data()``` method.
```python
from visikol_cnn.utils import Dataset

class ExampleDataset(Dataset):
  def load_data(self, path, input_shape):
    
    # Load training data here
    self.X["train"] = []
    self.y["train"] = []
    
    # Load training data here
    self.X["test"] = []
    self.y["test"] = []
    
    # Load validation data here (if any)
    self.X["validation"] = []
    self.y["validation"] = []
```

3. Create a main function that creates a ```Dataset```, ```Config```, and ```Model``` object calls the ```train()``` method of the ```Model``` object.

```python
from visikol_cnn.model import Model

if __name__ == "__main__":
  # Create the Config object
  config = ExampleConfig()
  
  # Create the Dataset object and load the data
  dataset = ExampleDataset()
  dataset.load_data()
    
  # Create the Model object and train
  model = Model()
  model.train(dataset, config)
  
  # Evaluate the test or validation data and view some predicitons
  model.evaluate_test(dataset, config, to_csv=True)
  model.visualize_class_predictions(dataset.X["validation"], dataset.y["validation"])
  
```
