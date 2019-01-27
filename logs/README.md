## logs
This is the default directroy for saving training logs and other output data. When training is completed the ```logs/``` directroy will have the following structure.

```
logs/
  save_name/
    date_string/
      config.pkl
      models/
        model_01.hdf5
      csv_files/
        training.csv
        results.csv
```   

**config.pkl**

The pickled ```Config``` object used during the training session.

**model.hdf5 files**

Keras model files which contain the model architecture and saved weights.

**training.csv**

Contains the loss values and evaluation metrics for the training data and validation data for each epoch.

**results.csv**

Contains the measured evaulation metrics for the test data. 
