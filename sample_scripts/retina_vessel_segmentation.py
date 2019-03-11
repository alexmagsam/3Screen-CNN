import os
import sys
import numpy as np
import PIL.Image as pil
from skimage.color import rgb2gray

sys.path.append('../')

from cnn.model import Model
from cnn.utils import Dataset
from cnn.config import Config


class RetinaConfig(Config):
    DATA_PATH = 'sample_data/retina-vessel-segmentation'
    INPUT_SHAPE = (256, 256, 1)
    SAVE_NAME = 'retina-unet-small-dice'
    MODEL_NAME = 'u-net-small'
    LOSS = 'dice'
    LEARNING_RATE = .0002
    OPTIMIZER = {"name": 'Adam', "momentum": 0, "decay": 0, "epsilon": 0}
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    NUM_CLASSES = 1
    VAL_SPLIT = .1


class RetinaDataset(Dataset):

    def load_data(self, path, input_shape, num_patches):
        # Load the training files
        image_files_train = next(os.walk(os.path.join(path, 'training', 'images')))[2]
        mask_files_train = next(os.walk(os.path.join(path, 'training', '1st_manual')))[2]

        self.X["all"] = np.zeros(((len(image_files_train)*num_patches, ) + input_shape), np.float32)
        self.y["all"] = np.zeros_like(self.X["all"])

        print('-' * 100)
        print("Loading training data.\n")

        for idx in range(len(image_files_train)):
            # Load the full size mask
            mask = np.array(pil.open(os.path.join(path, 'training', '1st_manual', mask_files_train[idx])))
            mask = np.expand_dims(mask.astype(np.float32), axis=2)

            # Load the full size image
            image = np.array(pil.open(os.path.join(path, 'training', 'images', image_files_train[idx])))
            image = np.expand_dims(rgb2gray(image), axis=2)

            # Extract random patches from the full size image and mask
            bbox_list = self.extract_random_positive_patches(mask, input_shape, num_patches=num_patches, positive_only=False)
            image_patches = self.extract_patches_from_list(image, bbox_list)
            mask_patches = self.extract_patches_from_list(mask, bbox_list)

            # Add the patches to the training set and rescale
            self.X["all"][idx*num_patches:(idx+1)*num_patches] = image_patches * 2 - 1
            self.y["all"][idx * num_patches:(idx + 1) * num_patches] = mask_patches / 255

        print("Loaded {} images and split into {} total patches.\n".format(len(image_files_train), self.X["all"].shape[0]))

        # Load the test files
        image_files_test = next(os.walk(os.path.join(path, 'test', 'images')))[2]
        mask_files_test = next(os.walk(os.path.join(path, 'test', '1st_manual')))[2]

        self.X["test"] = []
        self.y["test"] = []

        print('-' * 100)
        print("Loading testing data.\n")

        for idx in range(len(image_files_test)):
            # Load the full size mask
            mask = np.array(pil.open(os.path.join(path, 'test', '1st_manual', mask_files_test[idx])))
            mask = np.expand_dims(mask.astype(np.float32), axis=2)

            # Load the full size image
            image = np.array(pil.open(os.path.join(path, 'test', 'images', image_files_test[idx])))
            image = np.expand_dims(rgb2gray(image), axis=2)

            # Deconstruct the full size image and mask into patches
            image_patches = self.deconstruct_image(image, input_shape)
            mask_patches = self.deconstruct_image(mask, input_shape)

            # Add the patches to the test set and rescale
            self.X["test"].append(image_patches * 2 - 1)
            self.y["test"].append(mask_patches / 255)

        self.X["test"] = np.concatenate(self.X["test"], axis=0)
        self.y["test"] = np.concatenate(self.y["test"], axis=0)

        print("Loaded {} images and split into {} total patches.\n".format(len(image_files_test),
                                                                           self.X["test"].shape[0]))


if __name__ == '__main__':
    # Create the Config object
    config = RetinaConfig()

    # Create the Dataset object
    dataset = RetinaDataset()

    # Load the data and split 10% of the images from training for validation
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE, num_patches=250)
    dataset.split_data(validation_size=config.VAL_SPLIT)

    # Create the Model object
    model = Model()

    # Train the model
    model.train(dataset, config)

    # Evaulate the trained model and visualize some predictions made on the test set
    model.visualize_patch_segmentation_predictions(dataset.X["test"], dataset.y["test"])
