import os
import sys
import numpy as np
import PIL.Image as pil
import imgaug.augmenters as iaa
from skimage.transform import resize

sys.path.append('../')

from visikol_cnn.model import Model
from visikol_cnn.config import Config
from visikol_cnn.utils import Dataset


class GlandConfig(Config):
    DATA_PATH = r'E:\Deep Learning Datasets\Gland Segmentation'
    SAVE_NAME = 'gland-unet-small-bce'
    LOSS = 'bce'
    LEARNING_RATE = .0001
    OPTIMIZER = {'name': 'Adam', 'decay': 0, 'momentum': 0, 'epsilon': 0}
    BATCH_SIZE = 6
    NUM_EPOCHS = 20
    INPUT_SHAPE = (512, 768, 3)
    NUM_CLASSES = 1
    MODEL_NAME = 'u-net-small'


class GlandDataset(Dataset):

    testA_filenames = {'image': 'testA_{}.bmp', 'mask': 'testA_{}_anno.bmp', 'num_files': 60}
    testB_filenames = {'image': 'testB_{}.bmp', 'mask': 'testB_{}_anno.bmp', 'num_files': 20}
    train_filenames = {'image': 'train_{}.bmp', 'mask': 'train_{}_anno.bmp', 'num_files': 85}

    def load_data(self, path, input_shape):

        # Training Images
        print('-' * 100)
        print('Loading Training Images.\n')
        self.X["train"] = np.zeros(((self.train_filenames['num_files'], ) + input_shape), np.float32)
        self.y["train"] = np.zeros(((self.train_filenames['num_files'],) + input_shape[:2] + (1, )), np.float32)
        for idx in range(self.train_filenames['num_files']):
            bmp = np.array(pil.open(os.path.join(path, self.train_filenames['image'].format(idx + 1))), np.float32)
            self.X["train"][idx] = resize(bmp / 255 * 2 - 1, input_shape, preserve_range=True)
            mask = np.array(pil.open(os.path.join(path, self.train_filenames['mask'].format(idx + 1))))
            self.y["train"][idx] = resize(np.expand_dims(np.float32(mask > 0), 2), input_shape[:2] + (1,),
                                          preserve_range=True)

            print('Loaded {} of {}'.format(idx + 1, self.train_filenames['num_files']))

        # Validation Images
        print('-' * 100)
        print('Loading Validation Images.\n')
        self.X["validation"] = np.zeros(((self.testB_filenames['num_files'],) + input_shape), np.float32)
        self.y["validation"] = np.zeros(((self.testB_filenames['num_files'],) + input_shape[:2] + (1, )), np.float32)
        for idx in range(self.testB_filenames['num_files']):
            bmp = np.array(pil.open(os.path.join(path, self.testB_filenames['image'].format(idx + 1))), np.float32)
            self.X["validation"][idx] = resize(bmp / 255 * 2 - 1, input_shape, preserve_range=True)
            mask = np.array(pil.open(os.path.join(path, self.testB_filenames['mask'].format(idx + 1))))
            self.y["validation"][idx] = resize(np.expand_dims(np.float32(mask > 0), 2), input_shape[:2] + (1,),
                                               preserve_range=True)

            print('Loaded {} of {}'.format(idx + 1, self.testB_filenames['num_files']))

        # Testing Images
        print('-' * 100)
        print('Loading Testing Images.\n')
        self.X["test"] = np.zeros(((self.testA_filenames['num_files'],) + input_shape), np.float32)
        self.y["test"] = np.zeros(((self.testA_filenames['num_files'],) + input_shape[:2] + (1,)), np.float32)
        for idx in range(self.testA_filenames['num_files']):
            bmp = np.array(pil.open(os.path.join(path, self.testA_filenames['image'].format(idx + 1))), np.float32)
            self.X["test"][idx] = resize(bmp / 255 * 2 - 1, input_shape, preserve_range=True)
            mask = np.array(pil.open(os.path.join(path, self.testA_filenames['mask'].format(idx + 1))))
            self.y["test"][idx] = resize(np.expand_dims(np.float32(mask > 0), 2), input_shape[:2] + (1,),
                                         preserve_range=True)

            print('Loaded {} of {}'.format(idx + 1, self.testA_filenames['num_files']))

    def augment_training_data(self):

        print('-' * 100)
        print('Augmenting Training Data.\n')

        X = self.X["train"].copy()
        y = self.y["train"].copy()

        aug_X = [X]
        aug_y = [y]

        aug_X.append(np.rot90(X, k=2, axes=(1, 2)))
        aug_y.append(np.rot90(y, k=2, axes=(1, 2)))

        aug_X.append(np.flip(X, axis=2))
        aug_y.append(np.flip(y, axis=2))

        aug_X.append(np.flip(np.flip(X, axis=2), axis=1))
        aug_y.append(np.flip(np.flip(y, axis=2), axis=1))

        aug_X = np.vstack(aug_X)
        aug_y = np.vstack(aug_y)

        sequential = iaa.Sequential([iaa.Affine(rotate=(-20, 20))])
        seq = sequential.to_deterministic()
        rotated_X = seq.augment_images(aug_X)
        rotated_y = seq.augment_images(aug_y)

        sequential = iaa.Sequential([iaa.Affine(translate_px={"x": (-75, 75), "y": (-75, 75)})])
        seq = sequential.to_deterministic()
        translated_X = seq.augment_images(aug_X)
        translated_y = seq.augment_images(aug_y)

        self.X["train"] = np.concatenate((self.X["train"], translated_X, rotated_X), axis=0)
        self.y["train"] = np.concatenate((self.y["train"], translated_y, rotated_y), axis=0)
        
        shuffle = np.arange(len(self.X["train"]))
        np.random.shuffle(shuffle)

        self.X["train"] = self.X["train"][shuffle]
        self.y["train"] = self.y["train"][shuffle]


if __name__ == '__main__':

    # Create the Config object
    config = GlandConfig()

    # Create the Dataset object
    dataset = GlandDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE)

    # Augment the training data to create more samples
    dataset.augment_training_data()

    # Create the Model object
    model = Model()

    # Train the model
    model.train(dataset, config)

    # Evaluate the test data and view some predictions
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_patch_segmentation_predictions(dataset.X["test"], dataset.y["test"], num_predictions=1)

