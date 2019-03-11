import os
import sys
import numpy as np
import PIL.Image as pil
from skimage.transform import resize

sys.path.append('../')

from cnn.config import Config
from cnn.utils import Dataset
from cnn.model import Model


class UltrasoundConfig(Config):
    DATA_PATH = 'sample_data/ultrasound-nerve-segmentation'
    INPUT_SHAPE = (256, 256, 1)
    SAVE_NAME = 'ultrasound-unet-small-dice'
    LOSS = 'dice'
    OPTIMIZER = {"name": "Adam", "decay": 0, "momentum": 0.95, "epsilon": 0}
    LEARNING_RATE = .0001
    BATCH_SIZE = 16
    NUM_CLASSES = 1
    NUM_EPOCHS = 20
    MODEL_NAME = 'u-net-small'
    TEST_SPLIT = .1
    VAL_SPLIT = .1


class UltrasoundDataset(Dataset):

    def load_data(self, path, input_shape, nonzero_only=False):

        # Load all of the images

        print('-'*100)
        print("Loading Images\n")
        files = next(os.walk(path + '/train'))[2]
        files = np.array([file for file in files if '_mask' not in file])

        self.X["all"] = np.zeros(((len(files), ) + input_shape), np.float32)
        self.y["all"] = np.zeros(((len(files),) + input_shape), np.float32)

        for idx, file in enumerate(files):
            tif = np.array(pil.open(os.path.join(path + '/train', file)), np.float32)
            self.X["all"][idx] = resize(tif, input_shape, mode='constant', anti_aliasing=True,
                                        preserve_range=True) / 255 * 2 - 1

            tif = np.array(pil.open(os.path.join(path + '/train', file.split('.')[0] + '_mask.tif')), np.float32)
            self.y["all"][idx] = resize(tif, input_shape, mode='constant', anti_aliasing=True,
                                        preserve_range=True) / 255

            if (idx + 1) % 1000 == 0:
                print("Loaded {} of {}".format(idx + 1, len(files)))

        # Resizing the masks often creates values other than 0 or 1
        self.y["all"][self.y["all"] > 0] = 1

        # The option to train on non-empy masks only
        if nonzero_only:
            nonzero_samples = np.unique(np.nonzero(self.y["all"])[0])
            print("\nNumber of nonzero samples: ", nonzero_samples.size)

            self.X["all"] = self.X["all"][nonzero_samples]
            self.y["all"] = self.y["all"][nonzero_samples]


if __name__ == "__main__":

    # Create the Config object
    config = UltrasoundConfig()

    # Create the Dataset object and load the data
    dataset = UltrasoundDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE, nonzero_only=False)
    dataset.split_data(test_size=config.TEST_SPLIT, validation_size=config.VAL_SPLIT)

    # Create the model and train the model
    model = Model()
    model.train(dataset, config)

    # Evaluate the model on the test data and visualize some predictions
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_patch_segmentation_predictions(dataset.X["validation"], dataset.y["validation"], num_predictions=3)
