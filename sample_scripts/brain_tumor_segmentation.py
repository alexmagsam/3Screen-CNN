import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append('../')

from am_cnn.config import Config
from am_cnn.utils import Dataset
from am_cnn.model import Model


class BrainTumorConfig(Config):
    DATA_PATH = r'E:\Deep Learning Datasets\brain-tumor-segmentation\MICCAI_BraTS_2018_Data_Training'
    INPUT_SHAPE = (240, 240, 1)
    LOGS_DIR = '../logs'
    SAVE_NAME = 'braintumor-unet-small-dice'
    LOSS = 'dice'
    LEARNING_RATE = .0005
    OPTIMIZER = {"name": 'sgd', 'momentum': .99, 'decay': 0}
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    NUM_EPOCHS = 10
    MODEL_NAME = 'u-net-small'
    TEST_SPLIT = .2
    VAL_SPLIT = .2


class BrainTumorDataset(Dataset):

    filenames = ['\{}_seg.nii.gz', '\{}_flair.nii.gz', '\{}_t1.nii.gz', '\{}_t1ce.nii.gz', '\{}_t2.nii.gz']

    def load_data(self, path, input_shape, num_total):
        hgg_folders = next(os.walk(path + '\HGG'))[1]
        hgg_folders = [os.path.join(path + '\HGG', folder) for folder in hgg_folders]
        lgg_folders = next(os.walk(path + '\LGG'))[1]
        lgg_folders = [os.path.join(path + '\LGG', folder) for folder in lgg_folders]
        folders = lgg_folders + hgg_folders

        self.X["all"] = np.zeros(((1, ) + input_shape), np.float32)
        self.y["all"] = np.zeros((1, input_shape[0], input_shape[1], 4), np.float32)
        for folder in folders:
            print("\nLoading folder {}".format(folder.split('\\')[-1]))
            for idx, file in enumerate(self.filenames):
                if idx == 0:
                    nii = nib.load(folder + file.format(folder.split('\\')[-1]))
                    mask = nii.get_fdata()
                    slices = np.unique(np.transpose(np.nonzero(mask))[:, 2])
                    tmp = np.zeros((len(slices) * 4, mask.shape[0], mask.shape[1], 1), np.float32)
                    tmp_mask = np.zeros((len(slices) * 4, mask.shape[0], mask.shape[1], 4), np.float32)

                    for slice_idx, _slice in enumerate(slices):
                        for i in range(4):
                            for label in range(4):
                                tmp_mask[len(slices)*i+slice_idx, ..., label] = (mask[..., _slice] == (label + 1)).astype(np.float32)
                else:
                    nii = nib.load(folder + file.format(folder.split('\\')[-1]))
                    img = nii.get_fdata()
                    tmp[(idx - 1)*len(slices):idx*len(slices)] = np.expand_dims(np.rollaxis(img[..., slices], 2, 0), 3)

            self.X["all"] = np.concatenate((self.X["all"], tmp), axis=0)
            self.y["all"] = np.concatenate((self.y["all"], tmp_mask), axis=0)

            # Once enough the desired amount of images have been loaded, quit the loop
            print("{} images have been loaded into X and y".format(len(self.X["all"])))
            if len(self.X["all"]) > num_total:
                break

            # For debugging purposes to make sure the masks are in the right order
            for i in range(4):
                for slice_idx1, _slice1 in enumerate(slices):
                    for label in range(4):
                        equal = np.array_equal(tmp_mask[len(slices)*i+slice_idx1, ..., label] == 1,
                                               mask[..., _slice1] == label + 1)
                        if not equal:
                            plt.subplot(121)
                            plt.imshow(mask[..., _slice1] == label + 1, cmap='gray')
                            plt.subplot(122)
                            plt.imshow(tmp_mask[len(slices)*i+slice_idx1, ..., label] == 1, cmap='gray')
                            plt.show()
                            raise ValueError("There is not agreement at slice {}, index {}, multiplier {} and label {}".
                                             format(_slice1, slice_idx1, i, label+1))

        print("\nTotal number of images {}".format(len(self.X["all"])))
        print("Max in X = {}".format(self.X["all"].max()))
        print("Max in y = {}".format(self.y["all"].max()))

        self.X["all"] = self.X["all"][1:]
        self.y["all"] = self.y["all"][1:]

        rand_shuffle = np.arange(len(self.X["all"]))
        np.random.shuffle(rand_shuffle)

        self.X["all"] = self.X["all"][rand_shuffle] / self.X["all"].max()
        self.y["all"] = self.y["all"][rand_shuffle]


if __name__ == "__main__":

    # Create the Config object
    config = BrainTumorConfig()

    # Create the Dataset object and load the data
    dataset = BrainTumorDataset()
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE, 7500)
    dataset.split_data(test_size=config.TEST_SPLIT, validation_size=config.VAL_SPLIT)

    # Create the model and train the model
    model = Model()
    model.train(dataset, config)

