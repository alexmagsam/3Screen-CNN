import os
import sys
import numpy as np
import PIL.Image as pil
import nibabel as nib

sys.path.append('../')

from cnn.config import Config
from cnn.utils import Dataset
from cnn.model import Model


class BrainTumorConfig(Config):
    DATA_PATH = r'A:\Deep Learning Datasets Temp\Brain Tumor Segmentation'
    INPUT_SHAPE = (240, 240, 1)
    SAVE_NAME = 'braintumor-unet-small-dice'
    LOSS = 'dice'
    LEARNING_RATE = .0002
    OPTIMIZER = {"name": 'Adam', 'momentum': 0, 'decay': 0, 'epsilon': 0}
    BATCH_SIZE = 16
    NUM_CLASSES = 1
    NUM_EPOCHS = 30
    MODEL_NAME = 'u-net-small'
    NUM_TRAIN = 10000
    TEST_SPLIT = .1
    VAL_SPLIT = .1


class BrainTumorDataset(Dataset):

    filenames = ['{}_seg.nii.gz', '{}_flair.nii.gz', '{}_t1.nii.gz', '{}_t1ce.nii.gz', '{}_t2.nii.gz']

    def export_data(self, path):
        print('-'*100)
        print('Exporting and transforming images first.')

        # Make the export folder
        export_path = os.path.join(path, 'exported_images')
        image_path = os.path.join(export_path, 'image')
        mask_path = os.path.join(export_path, 'mask')
        try:
            os.mkdir(export_path)
            os.mkdir(image_path)
            os.mkdir(mask_path)
        except FileExistsError:
            pass

        if os.listdir(image_path) or os.listdir(mask_path):
            print('\nExport directories are not empty. Skipping this step.')
            return

        # Get all of the folder names containing the data
        hgg_folders = next(os.walk(os.path.join(path, 'HGG')))[1]
        hgg_folders = [os.path.join(path, 'HGG', folder) for folder in hgg_folders]
        lgg_folders = next(os.walk(os.path.join(path, 'LGG')))[1]
        lgg_folders = [os.path.join(path, 'LGG', folder) for folder in lgg_folders]
        folders = np.array(lgg_folders + hgg_folders)

        # Filename structure
        image_filename = '{}_{}_slice{:03d}.tif'
        mask_filename = '{}_{}_slice{:03d}_label{:02d}.tif'

        # Export all of the images and corresponding labels as TIF images
        print('-' * 100)
        for folder in folders:
            sample_name = os.path.split(folder)[1]
            print("Exporting sample {}".format(sample_name))
            for idx, file in enumerate(self.filenames):
                if idx == 0:
                    modality = file.split('_')[-1].split('.')[0]
                    nii = nib.load(os.path.join(folder, file.format(sample_name)))
                    mask = nii.get_fdata()
                    slices = np.unique(np.nonzero(mask)[2])
                    mask = mask[..., slices]
                    for slice_idx, _slice in enumerate(slices):
                        for label in range(4):
                            tif = pil.fromarray(np.uint8(mask[..., slice_idx] == label + 1)*255)
                            tif.save(os.path.join(mask_path, mask_filename.format(sample_name, modality, _slice, label + 1)))
                else:
                    modality = file.split('_')[-1].split('.')[0]
                    nii = nib.load(os.path.join(folder, file.format(sample_name)))
                    img = nii.get_fdata()
                    img = img[..., slices]
                    for slice_idx, _slice in enumerate(slices):
                        tif = pil.fromarray(np.uint8(img[..., slice_idx] / img[..., slice_idx].max() * 255))
                        tif.save(os.path.join(image_path, image_filename.format(sample_name, modality, _slice)))

    def load_data(self, path, input_shape, num_total):
        print('-'*100)
        print('Loading data.\n')
        # Paths to load the data from
        image_path = os.path.join(path, 'exported_images', 'image')
        mask_path = os.path.join(path, 'exported_images', 'mask')

        # Pick random files for training
        image_files = np.array(next(os.walk(image_path))[2])
        rand_idx = np.random.randint(0, len(image_files), num_total)
        image_files = image_files[rand_idx]

        # Create the input and output arrays for all of the data
        self.X["all"] = np.zeros(((len(image_files), ) + input_shape), np.float32)
        self.y["all"] = np.zeros(((len(image_files),) + input_shape), np.float32)

        for idx in range(len(image_files)):

            # Load the MRI scan
            tif = np.array(pil.open(os.path.join(image_path, image_files[idx])), np.float32)
            self.X["all"][idx] = np.expand_dims(tif / 255 * 2 - 1, 2)

            # Load the ground truth segmentation
            split_filename = image_files[idx].split('_')
            mask = np.zeros((input_shape[0], input_shape[1]), np.bool)
            for label in range(4):
                filename = '{}_{}_{}_{}_seg_{}_label{:02d}.tif'.format(split_filename[0], split_filename[1],
                                                                       split_filename[2], split_filename[3],
                                                                       split_filename[-1].split('.')[0], label + 1)
                tif = np.array(pil.open(os.path.join(mask_path, filename)), np.bool)
                mask = np.logical_or(mask, tif)

            self.y["all"][idx] = np.expand_dims(mask.astype(np.float32), 2)

            if (idx + 1) % 1000 == 0:
                print('Loaded {} of {}'.format(idx + 1, len(image_files)))

        print('\nX shape = ', self.X["all"].shape)
        print('y shape = ', self.y["all"].shape)

        print('\nX min = ', self.X["all"].min())
        print('X max = ', self.X["all"].max())

        print('\ny min = ', self.y["all"].min())
        print('y max = ', self.y["all"].max())


if __name__ == "__main__":

    # Create the Config object
    config = BrainTumorConfig()

    # Create the Dataset object
    dataset = BrainTumorDataset()

    # The data must be reformatted and exported for easier loading
    dataset.export_data(config.DATA_PATH)

    # Load the data
    dataset.load_data(config.DATA_PATH, config.INPUT_SHAPE, config.NUM_TRAIN)
    dataset.split_data(validation_size=config.VAL_SPLIT)

    # Create the model and train the model
    model = Model()
    model.train(dataset, config)

    # Evaluate the model on the test data and visualize some predictions
    model.evaluate_test(dataset, config, to_csv=True)
    model.visualize_patch_segmentation_predictions(dataset.X["validation"], dataset.y["validation"], num_predictions=1)
