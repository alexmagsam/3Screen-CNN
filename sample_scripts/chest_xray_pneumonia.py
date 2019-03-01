import os
import sys
import numpy as np
import PIL.Image as pil
from skimage.transform import resize
from skimage.color import gray2rgb
sys.path.append('../')

from visikol_cnn.config import Config
from visikol_cnn.utils import Dataset
from visikol_cnn.model import Model


class XRayConfig(Config):
    DATA_PATH = 'A:\Deep Learning Datasets Temp\Pneumonia Detection'
    INPUT_SHAPE = (512, 512, 3)
    SAVE_NAME = 'xray-inceptionv3-bce'
    LOSS = 'bce'
    OPTIMIZER = {"name": "Adam", "decay": 0, "momentum": 0, "epsilon": 0}
    LEARNING_RATE = .0001
    BATCH_SIZE = 8
    NUM_CLASSES = 1
    NUM_EPOCHS = 10
    MODEL_NAME = 'inceptionv3'


class XRayDataset(Dataset):

    class_names = ["normal", "pneumonia"]

    def load_data(self, path, input_shape):
        # Load the training data
        train_normal_files = next(os.walk(os.path.join(path, 'chest_xray', 'train', 'NORMAL')))[2]
        train_normal_files = [os.path.join(path, 'chest_xray', 'train', 'NORMAL', file) for file in train_normal_files]
        train_normal_labels = [0 for _ in range(len(train_normal_files))]

        train_pneumonia_files = next(os.walk(os.path.join(path, 'chest_xray', 'train', 'PNEUMONIA')))[2]
        train_pneumonia_files = [os.path.join(path, 'chest_xray', 'train', 'PNEUMONIA', file) for file in train_pneumonia_files]
        train_pneumonia_labels = [1 for _ in range(len(train_pneumonia_files))]

        files = np.array(train_normal_files + train_pneumonia_files)
        self.y["train"] = np.array(train_normal_labels + train_pneumonia_labels, np.float32)
        self.X["train"] = np.zeros(((len(files), ) + input_shape), np.float32)

        print('-' * 100)
        print('Loading training data\n')
        for idx, file in enumerate(files):
            self.X["train"][idx] = resize(gray2rgb(np.array(pil.open(file), np.float32) / 255 * 2 - 1),
                                          output_shape=input_shape, anti_aliasing=True, preserve_range=True)

            if (idx + 1) % 100 == 0:
                print("Loading training {} of {}".format(idx + 1, len(files)))

        random_shuffle = np.arange(len(files))
        np.random.shuffle(random_shuffle)

        self.X["train"] = self.X["train"][random_shuffle]
        self.y["train"] = self.y["train"][random_shuffle]

        # Load the validation data
        val_normal_files = next(os.walk(os.path.join(path, 'chest_xray', 'val', 'NORMAL')))[2]
        val_normal_files = [os.path.join(path, 'chest_xray', 'val', 'NORMAL', file) for file in val_normal_files]
        val_normal_labels = [0 for _ in range(len(val_normal_files))]

        val_pneumonia_files = next(os.walk(os.path.join(path, 'chest_xray', 'val', 'PNEUMONIA')))[2]
        val_pneumonia_files = [os.path.join(path, 'chest_xray', 'val', 'PNEUMONIA', file) for file in val_pneumonia_files]
        val_pneumonia_labels = [1 for _ in range(len(val_pneumonia_files))]

        files = np.array(val_normal_files + val_pneumonia_files)
        self.y["validation"] = np.array(val_normal_labels + val_pneumonia_labels, np.float32)
        self.X["validation"] = np.zeros(((len(files),) + input_shape), np.float32)

        print('-' * 100)
        print('Loading validation data\n')
        for idx, file in enumerate(files):
            self.X["validation"][idx] = resize(gray2rgb(np.array(pil.open(file), np.float32) / 255 * 2 - 1),
                                               output_shape=input_shape, anti_aliasing=True, preserve_range=True)

            if (idx + 1) % 2 == 0:
                print("Loading validation {} of {}".format(idx + 1, len(files)))

        random_shuffle = np.arange(len(files))
        np.random.shuffle(random_shuffle)

        self.X["validation"] = self.X["validation"][random_shuffle]
        self.y["validation"] = self.y["validation"][random_shuffle]

        # Load the test data
        test_normal_files = next(os.walk(os.path.join(path, 'chest_xray', 'test', 'NORMAL')))[2]
        test_normal_files = [os.path.join(path, 'chest_xray', 'test', 'NORMAL', file) for file in test_normal_files]
        test_normal_labels = [0 for _ in range(len(test_normal_files))]

        test_pneumonia_files = next(os.walk(os.path.join(path, 'chest_xray', 'test', 'PNEUMONIA')))[2]
        test_pneumonia_files = [os.path.join(path, 'chest_xray', 'test', 'PNEUMONIA', file) for file in test_pneumonia_files]
        test_pneumonia_labels = [1 for _ in range(len(test_pneumonia_files))]

        files = np.array(test_normal_files + test_pneumonia_files)
        self.y["test"] = np.array(test_normal_labels + test_pneumonia_labels, np.float32)
        self.X["test"] = np.zeros(((len(files),) + input_shape), np.float32)

        print('-' * 100)
        print('Loading test data\n')
        for idx, file in enumerate(files):
            self.X["test"][idx] = resize(gray2rgb(np.array(pil.open(file), np.float32) / 255 * 2 - 1),
                                         output_shape=input_shape, anti_aliasing=True, preserve_range=True)

            if (idx + 1) % 100 == 0:
                print("Loading test {} of {}".format(idx + 1, len(files)))

        random_shuffle = np.arange(len(files))
        np.random.shuffle(random_shuffle)

        self.X["test"] = self.X["test"][random_shuffle]
        self.y["test"] = self.y["test"][random_shuffle]


if __name__ == "__main__":

    # Create the Config object
    config = XRayConfig()

    # Create the Dataset object and load the data
    dataset = XRayDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE)

    # Create the Model object and train the model
    model = Model()
    model.train(dataset, config)

    # Evaluate the model on the test data and visualize some predictions
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_class_predictions(dataset.X["test"], dataset.y["test"], class_names=dataset.class_names)
