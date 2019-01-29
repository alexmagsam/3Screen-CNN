import os
import sys
import numpy as np
import pandas as pd
import PIL.Image as pil
import matplotlib.pyplot as plt
from skimage.transform import resize

sys.path.append('../')

from am_cnn.config import Config
from am_cnn.utils import Dataset
from am_cnn.model import Model


class SkinCancerConfig(Config):
    DATA_PATH = 'E:\Deep Learning Datasets\skin-cancer-mnist'
    INPUT_SHAPE = (256, 256, 3)
    LOGS_DIR = '../logs'
    SAVE_NAME = 'skincancer-vgg16-cce'
    LOSS = 'cce'
    LEARNING_RATE = .001
    BATCH_SIZE = 16
    NUM_CLASSES = 7
    NUM_EPOCHS = 1
    MODEL_NAME = 'vgg16'
    TEST_SPLIT = .2
    VAL_SPLIT = .2


class SkinCancerDataset(Dataset):

    label_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    def load_data(self, path, input_shape):
        df = pd.read_csv(os.path.join(path, 'HAM10000_metadata.csv'))
        files = np.array(df['image_id'].tolist())
        labels = np.array(df['dx'].tolist())

        self.X["all"] = np.zeros(((len(files), ) + input_shape), np.float32)
        self.y["all"] = np.zeros((len(files), len(self.label_names)), np.float32)
        for idx, file in enumerate(files):
            if os.path.exists(os.path.join(path, 'HAM10000_images_part_1', file + '.jpg')):
                jpg = resize(np.array(pil.open(os.path.join(path, 'HAM10000_images_part_1', file + '.jpg'))) / 255,
                             input_shape, anti_aliasing=True, preserve_range=True)
                self.X["all"][idx] = jpg.copy()
                self.y["all"][idx, self.label_map[labels[idx]]] = 1
            elif os.path.exists(os.path.join(path, 'HAM10000_images_part_2', file + '.jpg')):
                jpg = resize(np.array(pil.open(os.path.join(path, 'HAM10000_images_part_2', file + '.jpg'))) / 255,
                             input_shape, anti_aliasing=True, preserve_range=True)
                self.X["all"][idx] = jpg.copy()
                self.y["all"][idx, self.label_map[labels[idx]]] = 1
            else:
                raise FileNotFoundError

            if (idx + 1) % 1000 == 0:
                print("Loaded {} of {}".format(idx + 1, len(files)))

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