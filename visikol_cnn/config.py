import os
import datetime
import pickle


class Config:
    """
    A parent class that contains configurations parameters crucial for training a CNN.

    ...

    Attributes
    ----------
    DATA_PATH : str
        A file directory that contains the data for training a CNN.
    SAVE_NAME : str
        A name for the training session. Use the format 'dataset_model_loss'.
    LOGS_DIR : str
        A file directory to save the training logs in.
    DATE_STRING : str
        A string that contains the date and time when training begins. Generated automatically, do not override.
    TRAINING_DIR : str
        A sub-directory that is created in the logs directory named after the SAVE_NAME attribute.
    CSV_DIR : str
        A sub-directory in the training directory where csv files are stored.
    MODELS_DIR : str
        A sub-directory in the training directory where the model files are stored.
    WEIGHTS : str
        The weights to begin training. Options are 'imagenet' or None. Imagenet weights can only be used with popular
        network architectures.
    NUM_CLASSES : int
        The number of classes used in training.
    BATCH_SIZE : int
        The number of images to use in each batch during training.
    LEARNING_RATE : float
        The learning rate to use during training.
    LR_SCHEDULER : bool
        Tells whether to use the learning rate scheduler function or not.
    LOSS : str
        The loss function to use during training. Valid options are:
            'bce' : binary cross-entropy, used for binary classification or semantic segmentation.
            'dice' : Dice loss, which is based on the Dice coefficient used for semantic segmentation.
            jaccard' : Jaccard loss, which is based on the Jaccard index used for semantic segmentation.
    OPTIMIZER : dict
        A dictionary the describes what optimizer to use along with parameters of the optimizer.
    NUM_EPOCHS : int
        The number of epochs during training.
    INPUT_SHAPE : tuple, (height, width, n_classes or n_channels) or (height, width, n_channels, n_classes)
        Describes the shape input tensor to the input layer of the CNN.
    IMG_SHAPE : tuple, (height, width, n_channels) or (height, width)
        Describes the shape of the image data that is uploaded before training, if the shape is the same among all images.
    NUM_TRAIN : int
        Number of training samples to create during augmentation.
    TEST_SPLIT : float
        Ratio of data to split from the entire set of images to use as test data.
    VAL_SPLIT : float
        Ratio of data to split from the entire set of images to use as validation data.
    MODEL_NAME : str
        The name of the CNN model to use. Valid options are:
            'u-net' : A popular CNN for semantic segmentation, especially with biomedical images. Works well for a
                      small training set.
            'inceptionv3' : A popular CNN for image classification and a top performer in the Imagenet challenge.
                            Requires a fairly large data set to prevent over-fitting.

    """

    DATA_PATH = None

    SAVE_NAME = None

    LOGS_DIR = 'logs/'

    DATE_STRING = None

    TRAINING_DIR = None

    CSV_DIR = None

    MODELS_DIR = None

    WEIGHTS = 'imagenet'

    NUM_CLASSES = 1

    BATCH_SIZE = 32

    LEARNING_RATE = .001

    LR_SCHEDULER = False

    LOSS = 'bce'

    OPTIMIZER = {"name": 'Adam', "decay": 0, "momentum": 0, "epsilon": 0}

    NUM_EPOCHS = 20

    INPUT_SHAPE = (64, 64, 1)

    IMG_SHAPE = (None, None)

    NUM_TRAIN = 1000

    TEST_SPLIT = 0.2

    VAL_SPLIT = 0.2

    MODEL_NAME = 'U-net'

    def __init__(self):
        x = datetime.datetime.now()
        self.DATE_STRING = "{}-{}-{}_{}.{}.{}".format(x.strftime('%m'), x.strftime('%d'), x.strftime('%Y'),
                                                      x.strftime('%H'), x.strftime('%M'), x.strftime('%S'))

    def create_training_directories(self):
        if not os.path.exists(self.LOGS_DIR):
            os.mkdir(self.LOGS_DIR)
        if not os.path.exists(os.path.join(self.LOGS_DIR, self.SAVE_NAME)):
            os.mkdir(os.path.join(self.LOGS_DIR, self.SAVE_NAME))
        if not os.path.exists(os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING)):
            self.TRAINING_DIR = os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING)
            os.mkdir(self.TRAINING_DIR)
        if not os.path.exists(os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING, 'models')):
            self.MODELS_DIR = os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING, 'models')
            os.mkdir(self.MODELS_DIR)
        if not os.path.exists(os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING, 'csv_files')):
            self.CSV_DIR = os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING, 'csv_files')
            os.mkdir(self.CSV_DIR)

        # Save the config object as pickle for later functionality
        with open(os.path.join(self.LOGS_DIR, self.SAVE_NAME, self.DATE_STRING, 'config.pkl'), 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


