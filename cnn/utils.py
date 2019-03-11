import numpy as np
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from keras import backend as K


class Dataset:

    X = {"all": np.array([]), "full": np.array([]), "train": np.array([]), "test": np.array([]), "validation": np.array([])}
    y = {"all": np.array([]), "full": np.array([]), "train": np.array([]), "test": np.array([]), "validation": np.array([])}
    label_map = {}
    class_names = []

    """ A parent class used to store, organize, augment data used to train a CNN.

    Attributes
    ----------
    X: dict
        A dictionary used to hold input data with keys:
            'all' : ndarray
                An array that holds all of the input images and can be split into training, testing, and validation data
                using the split_data() method.
            'full' : ndarray
                If the input image shape is larger the shape of the input layer of the CNN, this key is used to store
                the full size images until they are split into smaller patches.
            'train' : ndarray
                An array that stores the training images.
            'test' : ndarray
                An array that stores the test images.
            'validation' : ndarray
                An array that stores the validation images.
    y: dict
        A dictionary used to hold ground truth data with keys:
            'all' : ndarray
                An array that holds all of the ground truth data and can be split into training, testing, and validation
                data using the split_data() method.
            'full' : ndarray
                If the ground truth data shape is larger the shape of the output layer of the CNN, this key is used to
                store the full size data until they are split into smaller patches.
            'train' : ndarray
                An array that stores the ground truth data for training.
            'test' : ndarray
                An array that stores the ground truth data for testing.
            'validation' : ndarray
                An array that stores the ground truth data for validation.
    label_map : dict
        A dictionary that uses class names as keys and integers as values to represent classes. Ex. {'dog':0, 'cat':1}
    class_names : list
        A list of class names. Order matters, each class name must be at the index that matches the enumerated class value.

    Methods
    -------
    load_data(**kwargs)
        An abstract method that is meant to be overridden based on the specific that is being used.
    augment_data(X, y=None, seq=None)
        A static method that can be used to augment data for training.
    split_data(test_size=None, validation_size=None)
        A method that divides X["all"] and y["all"] into training, testing, and validation data.
    deconstruct_image(img, patch_shape)
        A static method that is used to break down a larger image into smaller patches.
    reconstruct_image(patch, image_shape)
        A static method used to reconstruct a larger image from smaller patches.
    extract_random_positive_patches(mask, patch_shape, num_patches, threshold=False, img=None, positive_only=True)
        A static method to extract image patches or patch coordinates from a larger image.
    extract_patches_from_list(img, bbox_list)
        A static method to extract image patches from a larger image.

    """

    def __init__(self):
        pass

    def load_data(self, **kwargs):
        """A method to load data into the X and y attributes of the subclass.

        Parameters
        ----------
        **kwargs : dict
            A dictionary containing parameters specific to the overridden function for loading data.
        """
        pass

    @staticmethod
    def augment_data(X, y=None, seq=None):
        """Static method to augment training data.

        Important note: Use unsigned 8-bit format, this format is supported much better. Float values often yield
        strange results.

        Parameters
        ----------
        X : ndarray
            Training images with shape [n_samples, height, width, n_channels] of 8-bit unsigned integer type.
        y : ndarray, None
            Training masks that accompany the training images with shape [n_samples, height, width, n_channels] of 8-bit
            unsigned integer type.
        seq : imgaug Sequential object
            A custom imgaug Sequential augmentation pipeline.

        Returns
        -------
        ndarray
            Augmented training input data.
        ndarray
            Augmented training ground truth data.

        """

        if X.dtype != np.uint8:
            raise TypeError("Input images must be 8-bit unsigned integer type.")

        # Create the default augmentation sequences
        if seq is None and y is None:
            seq = iaa.Sequential([iaa.Fliplr(.5), iaa.ContrastNormalization((.8, 1.2)),
                                  iaa.Add((-10, 10)), iaa.Multiply((.8, 1.1)), iaa.AdditiveGaussianNoise(scale=(0, 5))])
        if seq is None and y is not None:
            seq = iaa.Sequential([iaa.Fliplr(.5), iaa.Flipud(.5), iaa.Affine(translate_px=(-5, 5))])
        seq_det = seq.to_deterministic()

        # Augment the data
        aug_X = seq_det.augment_images(X)
        if y is not None:
            aug_y = seq_det.augment_images(y)
            return aug_X, aug_y
        else:
            return aug_X

    def split_data(self, test_size=None, validation_size=None):
        """Splits all of the data into training/testing or training/testing/validation data

        Parameters
        ----------
        test_size : float
            A ratio of amount of testing data to all of the data.
        validation_size : float, optional
            A ratio of amount of validation data to all of the data.

        """
        split = False
        if validation_size is None and test_size is not None:
            self.X["train"], self.X["test"], self.y["train"], self.y["test"] = train_test_split(self.X["all"],
                                                                                                self.y["all"],
                                                                                                test_size=test_size)
            split = True
        elif validation_size is not None and test_size is None:
            self.X["train"], self.X["validation"], self.y["train"], self.y["validation"] = \
                train_test_split(self.X["all"], self.y["all"], test_size=validation_size)
            split = True
        elif validation_size is not None and test_size is not None:
            self.X["train"], X_remainder, self.y["train"], y_remainder = \
                train_test_split(self.X["all"], self.y["all"], test_size=test_size+validation_size)
            self.X["test"], self.X["validation"], self.y["test"], self.y["validation"] = \
                train_test_split(X_remainder, y_remainder, test_size=validation_size/(validation_size+test_size))
            split = True
        else:
            print("No data has been split.")

        if split:
            self.X["all"] = np.array([])
            self.y["all"] = np.array([])

    @staticmethod
    def deconstruct_image(img, patch_shape):
        """Deconstructs a single image into several smaller patches.

        Parameters
        ----------
        img : ndarray
            Array of shape [height, width, n_channels].
        patch_shape : ndarray or tuple
            Array of shape [height, width, n_channels]

        Returns
        -------
        patch : ndarray
            Array of shape [n_patches, height, width, n_channels or n_classes]

        """
        pad_y = 0 if img.shape[0] % patch_shape[0] == 0 else patch_shape[0] - img.shape[0] % patch_shape[0]
        pad_x = 0 if img.shape[1] % patch_shape[1] == 0 else patch_shape[1] - img.shape[1] % patch_shape[1]
        img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), 'constant')

        n_y = int(img.shape[0] / patch_shape[0])
        n_x = int(img.shape[1] / patch_shape[1])

        patch = np.zeros((n_x*n_y, patch_shape[0], patch_shape[1], patch_shape[2]))

        for i in range(n_y):
            for j in range(n_x):
                x1 = j*patch_shape[1]
                x2 = (j+1)*patch_shape[1]
                y1 = i * patch_shape[0]
                y2 = (i + 1) * patch_shape[0]
                patch[i*n_x+j, :, :, :] = img[y1:y2, x1:x2, :]

        return patch

    @staticmethod
    def reconstruct_image(patch, image_shape):
        """Reconstructs several patches into a single image.

        Parameters
        ----------
        patch : ndarray
            Array of shape [n_patches, height, width, n_channels].
        image_shape : ndarray or tuple
            Array of shape [height, width, n_channels]

        Returns
        -------
        img : ndarray
            Array of shape [height, width, n_channels]

        """

        pad_y = 0 if image_shape[0] % patch.shape[1] == 0 else patch.shape[1] - image_shape[0] % patch.shape[1]
        pad_x = 0 if image_shape[1] % patch.shape[2] == 0 else patch.shape[2] - image_shape[1] % patch.shape[2]
        img = np.zeros((image_shape[0]+pad_y, image_shape[1]+pad_x, image_shape[2]))

        n_y = int(img.shape[0] / patch.shape[1])
        n_x = int(img.shape[1] / patch.shape[2])

        for i in range(n_y):
            for j in range(n_x):
                x1 = j*patch.shape[2]
                x2 = (j+1)*patch.shape[2]
                y1 = i * patch.shape[1]
                y2 = (i + 1) * patch.shape[1]
                img[y1:y2, x1:x2, :] = patch[i*n_x+j, :, :, :]

        return img[:image_shape[0], :image_shape[1], :]

    @staticmethod
    def extract_random_positive_patches(mask, patch_shape, num_patches, threshold=False, img=None, positive_only=True):
        """Extracts patches from an image only at locations where mask is True.

        Parameters
        ---------
        mask : ndarray
            Binary array of shape [height, width] where each pixel is represented by True or False.
        patch_shape : tuple
            Shape of the patches to be extracted of shape (height, width, n_channels).
        num_patches : int
            Number of positive patches to extract from the entire image.
        threshold : float, False
            A ratio of (True pixels / total pixels) that is used to determine if a patch is considered positive. If
            False, there only needs to be one true pixel.
        img : ndarray, None
            Image to extract the patches from.
        positive_only : bool, False
            Set False if you want to extract empty patches as well.

        Returns
        -------
        patch : ndarray or list
            Array positive patches of shape [num_patches, patch_shape[0], patch_shape[1], patch_shape[2]] if
            'img' is provided as input. Otherwise, a list of bounding boxes is output where each item in the list is
            a list [x1, y1, x2, y2].

        """

        if patch_shape[0] % 2 != 0 or patch_shape[1] % 2 != 0:
            raise ValueError("Only even dimensions are allowed")

        # Find indices where mask is True
        positive_idxs = np.transpose(mask.nonzero())

        # If an images is provided, return the extracted patches. Or else, returned a list of bounding boxes.
        if img is not None:
            patch = np.zeros((num_patches, patch_shape[0], patch_shape[1], img.shape[2]), np.float32)
        else:
            patch = []

        # Loop until num_patches has been extracted
        extracted_patches = 0
        while extracted_patches != num_patches:

            valid_bbox = False
            while not valid_bbox:

                if positive_only:
                    rand_idx = np.random.randint(0, positive_idxs.shape[0])
                    rand_pt = positive_idxs[rand_idx]
                else:
                    rand_pt = (np.random.randint(0, mask.shape[0]), np.random.randint(0, mask.shape[1]))

                x1 = rand_pt[1] - int(patch_shape[1] / 2)
                y1 = rand_pt[0] - int(patch_shape[0] / 2)
                x2 = rand_pt[1] + int(patch_shape[1] / 2)
                y2 = rand_pt[0] + int(patch_shape[0] / 2)

                if x1 < 0:
                    x1 = 0
                    x2 = x1 + patch_shape[1]
                elif x2 > mask.shape[1]:
                    x2 = mask.shape[1]
                    x1 = x2 - patch_shape[1]

                if y1 < 0:
                    y1 = 0
                    y2 = y1 + patch_shape[0]
                elif y2 > mask.shape[0]:
                    y2 = mask.shape[0]
                    y1 = y2 - patch_shape[0]

                if threshold:
                    if np.count_nonzero(mask[y1:y2, x1:x2]) / np.size(mask[y1:y2, x1:x2]) >= threshold:
                        valid_bbox = True
                else:
                    valid_bbox = True

                if valid_bbox:
                    if img is not None:
                        patch[extracted_patches] = img[y1:y2, x1:x2, :]
                    else:
                        patch.append([x1, y1, x2, y2])
                    extracted_patches += 1

        return patch

    @staticmethod
    def extract_patches_from_list(img, bbox_list):
        """ Extract smaller image patches from a larger image and a list of bounding boxes output from the
            extract_random_positive_patches() method.

        Parameters
        ----------
        img : ndarray
            Array of shape [height, width, n_channels]
        bbox_list : list
            List of lists where each sublist has the coordinates [x1, y1, x2, y2].

        Returns
        -------
        patch : ndarray
            Array of shape [len(bbox_list), patch_height, patch_width, n_channels]

        """

        bbox = bbox_list[0]
        patch = np.zeros((len(bbox_list), bbox[3]-bbox[1], bbox[2]-bbox[0], img.shape[2]), img.dtype)
        for idx, bbox in enumerate(bbox_list):
            patch[idx] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        return patch


#######################################################################################################################
# Loss functions and metrics

smooth = 1.


def precision_binary(y_true, y_pred):
    """ Keras metric for computing precision for a binary classification task during training

    Precision = true positives / (true positives + false positives)

    Parameters
    ----------
    y_true : K.variable
        Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
    y_pred : K.variable
        Predicted Keras variable of float type with predicted values between 0 and 1.

    Returns
    -------
    K.variable
        A single value indicating the precision.

    """

    logical_and = K.cast(K.all(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    logical_or = K.cast(K.any(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    tp = K.sum(logical_and)
    fp = K.sum(logical_or - y_true)
    return K.switch(K.equal(tp, K.variable(0)), K.variable(0), tp / (tp + fp))


def recall_binary(y_true, y_pred):
    """ Keras metric for computing recall for a binary classification task during training

        Recall = true positives / (true positives + false negatives)

        Parameters
        ----------
        y_true : K.variable
            Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
        y_pred : K.variable
            Predicted Keras variable of float type with predicted values between 0 and 1.

        Returns
        -------
        K.variable
            A single value indicating the recall.

    """

    logical_and = K.cast(K.all(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    logical_or = K.cast(K.any(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    tp = K.sum(logical_and)
    fn = K.sum(logical_or - K.cast(K.greater_equal(y_pred, 0.5), 'float32'))
    return K.switch(K.equal(tp, K.variable(0)), K.variable(0), tp / (tp + fn))


def binary_crossentropy_np(y_true, y_pred):
    """Numpy metric for computing binary cross-entropy loss.

    Parameters
    ----------
    y_true : ndarray
        Ground truth Numpy array of float type with only 0's and 1's.
    y_pred : ndarray
        Predicted Numpy array of float type with predicted values between 0 and 1.

    Returns
    -------
    float
        A single value indicating the binary cross-entropy loss

    """
    y_pred = np.clip(y_pred, 1e-6, 1-1e-6)
    return -np.mean((y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))


def jaccard(y_true, y_pred):
    """ Keras metric for computing Jaccard index, also known as Intersection over Union (IoU).

    Jaccard index =   (A ∩ B) / (A ᴜ B)
                  =    TP / (TP + FP + FN)

    Parameters
    ----------
    y_true : K.variable
        Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
    y_pred : K.variable
        Predicted Keras variable of float type with predicted values between 0 and 1.

    Returns
    -------
    K.variable
        A single value indicating the Jaccard index.

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_ = K.sum(y_true_f + y_pred_f)
    return K.switch(K.equal(sum_ - intersection, K.variable(0)), K.variable(0), intersection / (sum_ - intersection))


def jaccard_loss(y_true, y_pred):
    """ Keras metric for computing Jaccard loss, also known as Intersection over Union (IoU).

    Parameters
    ----------
    y_true : K.variable
        Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
    y_pred : K.variable
        Predicted Keras variable of float type with predicted values between 0 and 1.

    Returns
    -------
    K.variable
        A single value indicating the Jaccard loss.

    """

    return -jaccard(y_true, y_pred)


def jaccard_np(y_true, y_pred):
    """Numpy metric for computing Jaccard index (see above).

    Parameters
    ----------
    y_true : ndarray
        Ground truth Numpy array of float type with only 0's and 1's.
    y_pred : ndarray
        Predicted Numpy array of float type with predicted values between 0 and 1.

    Returns
    -------
    float
        A single value indicating the Jaccard index.

    """

    intersection = np.sum(np.abs(y_pred) * y_true)
    sum_ = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))
    jc = intersection / (sum_ - intersection) if sum_ - intersection != 0 else 0
    return jc


def dice(y_true, y_pred):
    """Keras metric for computing Dice coefficient.

    Dice coefficient =  2*(A ∩ B) / (A + B)
                     =  2*TP / (2*TP + FP + FN)

    Parameters
    ----------
    y_true : K.variable
        Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
    y_pred : K.variable
        Predicted Keras variable of float type with predicted values between 0 and 1.

    Returns
    -------
    K.variable
        A single value indicating the Dice coefficient.

    References
    ----------
    https://github.com/jocicmarko/ultrasound-nerve-segmentation

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.switch(K.equal((K.sum(y_true_f) + K.sum(y_pred_f)), K.variable(0)), K.variable(0),
                    (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f)))


def dice_loss(y_true, y_pred):
    """Keras metric for computing Dice loss (see above).

    Parameters
    ----------
    y_true : K.variable
        Ground truth N-dimensional Keras variable of float type with only 0's and 1's.
    y_pred : K.variable
        Predicted Keras variable of float type with predicted values between 0 and 1.

    Returns
    -------
    K.variable
        A single value indicating the Dice loss.

    """
    return -dice(y_true, y_pred)


def dice_np(y_true, y_pred):
    """Numpy metric for computing Dice coefficient (see above).

    Parameters
    ----------
    y_true : ndarray
        Ground truth Numpy array of float type with only 0's and 1's.
    y_pred : ndarray
        Predicted Numpy array of float type with predicted values between 0 and 1.

    Returns
    -------
    float
        A single value indicating the Dice coefficient.

    """

    intersection = np.sum(np.abs(y_pred * y_true))
    sum_ = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))
    dc = (2 * intersection) / sum_ if sum_ != 0 else 0
    return dc


########################################################################################################################
# Miscellaneous


def training_summary_dict(dataset, config):
    return {"date": config.DATE_STRING, "num train": len(dataset.X["train"]), "num test": len(dataset.X["test"]),
            "num validation": len(dataset.X["validation"]), "model name": config.MODEL_NAME,
            'input height': config.INPUT_SHAPE[0], 'input width': config.INPUT_SHAPE[1],
            'input channels': config.INPUT_SHAPE[2], "loss": config.LOSS, "optimizer": config.OPTIMIZER["name"],
            "learning rate": config.LEARNING_RATE, "decay": config.OPTIMIZER["decay"],
            "momentum": config.OPTIMIZER["momentum"], "epsilon": config.OPTIMIZER["epsilon"],
            "num epochs": config.NUM_EPOCHS, "num classes": config.NUM_CLASSES}


########################################################################################################################
