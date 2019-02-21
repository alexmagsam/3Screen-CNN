import os
import sys
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import gray2rgb

sys.path.append('../')

from visikol_cnn.config import Config
from visikol_cnn.utils import Dataset
from visikol_cnn.model import Model


class UltrasoundConfig(Config):
    DATA_PATH = r'E:\Deep Learning Datasets\Ultrasound Nerve Segmentation'
    INPUT_SHAPE = (256, 256, 1)
    SAVE_NAME = 'ultrasound-unet-small-dice'
    LOSS = 'dice'
    OPTIMIZER = {"name": "RMSProp", "decay": 0.9, "momentum": 0.9, "epsilon": 0}
    LEARNING_RATE = .001
    BATCH_SIZE = 16
    NUM_CLASSES = 1
    NUM_EPOCHS = 5
    MODEL_NAME = 'u-net-small'
    TEST_SPLIT = .1
    VAL_SPLIT = .1


class UltrasoundDataset(Dataset):

    def load_data(self, path, input_shape, num_total):
        files = next(os.walk(path + '/train'))[2]
        files = np.array([file for file in files if '_mask' not in file])

        rand_idx = np.random.randint(0, len(files), num_total)
        files = files[rand_idx]

        self.X["all"] = np.zeros(((len(files), ) + input_shape), np.float32)
        self.y["all"] = np.zeros(((len(files),) + input_shape), np.float32)

        for idx, file in enumerate(files):
            tif = np.array(pil.open(os.path.join(path + '/train', files[0])), np.float32)
            self.X["all"][idx] = resize(tif, input_shape, mode='constant', anti_aliasing=True,
                                        preserve_range=True) / 255 * 2 - 1

            tif = np.array(pil.open(os.path.join(path + '/train', files[0].split('.')[0] + '_mask.tif')), np.float32)
            self.y["all"][idx] = resize(tif, input_shape, mode='constant', anti_aliasing=True,
                                        preserve_range=True) / 255

            if (idx + 1) % 1000 == 0:
                print("Loaded {} of {}".format(idx + 1, len(files)))

        self.y["all"][self.y["all"] > 0] = 1


if __name__ == "__main__":

    # Create the Config object
    config = UltrasoundConfig()

    # Create the Dataset object and load the data
    dataset = UltrasoundDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE, 5000)
    dataset.split_data(test_size=config.TEST_SPLIT, validation_size=config.VAL_SPLIT)

    # Create the model and train the model
    model = Model()
    model.train(dataset, config)
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_patch_segmentation_predictions(dataset.X["validation"], dataset.y["validation"])
