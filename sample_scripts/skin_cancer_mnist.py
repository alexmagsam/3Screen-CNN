import os
import sys
import numpy as np
import pandas as pd
import PIL.Image as pil
from skimage.transform import resize

sys.path.append('../')

from cnn.config import Config
from cnn.utils import Dataset
from cnn.model import Model


class SkinCancerConfig(Config):
    DATA_PATH = 'sample_data/skin-cancer-mnist'
    INPUT_SHAPE = (256, 256, 3)
    SAVE_NAME = 'skincancer-vgg16-cce'
    LOSS = 'cce'
    OPTIMIZER = {"name": "Adam", "decay": 0, "momentum": 0, "epsilon": 0}
    LEARNING_RATE = .0001
    BATCH_SIZE = 16
    NUM_CLASSES = 7
    NUM_EPOCHS = 20
    WEIGHTS = None
    MODEL_NAME = 'vgg16'
    TEST_SPLIT = .1
    VAL_SPLIT = .1


class SkinCancerDataset(Dataset):

    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    def load_data(self, path, input_shape):
        # Read the filenames from the CSV file
        df = pd.read_csv(os.path.join(path, 'HAM10000_metadata.csv'))
        files = np.array(df['image_id'].tolist())
        labels = np.array(df['dx'].tolist())

        self.X["all"] = np.zeros(((len(files), ) + input_shape), np.float32)
        self.y["all"] = np.zeros((len(files), len(self.class_names)), np.float32)

        print('-'*100)
        print('Loading images.\n')
        for idx, file in enumerate(files):
            if os.path.exists(os.path.join(path, 'HAM10000_images_part_1', file + '.jpg')):
                jpg = np.array(pil.open(os.path.join(path, 'HAM10000_images_part_1', file + '.jpg'))).astype(np.float32)
                self.X["all"][idx] = resize(jpg, input_shape, anti_aliasing=True, preserve_range=True) / 255 * 2 - 1
                self.y["all"][idx, self.label_map[labels[idx]]] = 1
            elif os.path.exists(os.path.join(path, 'HAM10000_images_part_2', file + '.jpg')):
                jpg = np.array(pil.open(os.path.join(path, 'HAM10000_images_part_2', file + '.jpg'))).astype(np.float32)
                self.X["all"][idx] = resize(jpg, input_shape, anti_aliasing=True, preserve_range=True) / 255 * 2 - 1
                self.y["all"][idx, self.label_map[labels[idx]]] = 1
            else:
                raise FileNotFoundError

            if (idx + 1) % 1000 == 0:
                print("Loaded {} of {}".format(idx + 1, len(files)))

        # Shuffle the images
        random_shuffle = np.arange(len(files))
        np.random.shuffle(random_shuffle)

        self.X["all"] = self.X["all"][random_shuffle]
        self.y["all"] = self.y["all"][random_shuffle]


if __name__ == "__main__":

    # Create the Config object
    config = SkinCancerConfig()

    # Create the Dataset object and load the data
    dataset = SkinCancerDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE)
    dataset.split_data(test_size=config.TEST_SPLIT, validation_size=config.VAL_SPLIT)

    # Create the Model object and train the model
    model = Model()
    model.train(dataset, config)

    # Evaluate the model on the test data and visualize some predictions
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_class_predictions(dataset.X["test"], dataset.y["test"], class_names=dataset.class_names)