import numpy as np
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from keras import backend as K
from skimage.measure import regionprops, label


class Dataset:
    """A parent class used to store, organize, augment data used to train a CNN.

    ...

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

    Methods
    -------
    load_data(**kwargs)
        An abstract method that is meant to be overridden based on the specific that is being used.
    augment_data(X, y, num_train, masks=False)
        A static method that can be used to augment data for training.
    deconstruct_image(img, patch_shape)
        A static method that is used to break down a larger image into smaller patches.
    reconstruct_image(patch, image_shape)
        A static method used to reconstruct a larger images form smaller patches.

    """

    X = {"all": np.array([]), "full": np.array([]), "train": np.array([]), "test": np.array([]), "validation": np.array([])}
    y = {"all": np.array([]), "train": np.array([]), "test": np.array([]), "validation": np.array([])}

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
    def augment_data(X, y, num_augment, masks=False):
        """Static method to augment training data.

        Parameters
        ----------
        X : ndarray
            Training images with shape [n_samples, height, width, n_channels or n_classes].
        y : ndarray
            Ground truth array with shape [n_samples, height, width, n_classes] or
            [n_samples, n_classes] or [n_samples].
        num_train : int
            Amount of desired training data.
        masks : bool
            Indicates whether the ground truth data are masks or not.

        Returns
        -------
        ndarray
            Augmented training input data.
        ndarray
            Augmented training ground truth data.

        """

        rand_repeated = np.random.randint(0, X.shape[0], num_augment)
        aug_X = X[rand_repeated]
        aug_y = y[rand_repeated]

        # Create the augmentation sequence
        # TODO: Augmentation needs to be adjust based in the data set.
        seq = iaa.Sequential([iaa.Fliplr(.5), iaa.Affine(rotate=(-10, 10)), iaa.Add((-20, 20)),
                              iaa.GaussianBlur(sigma=(0.0, 1.0)), iaa.ContrastNormalization((.75, 1.25))])
        seq_det = seq.to_deterministic()

        # Augment the data
        aug_X = seq_det.augment_images(aug_X)
        if masks:
            aug_y = seq_det.augment_images(aug_y)
        return np.concatenate((X, aug_X), 0), np.concatenate((y, aug_y), 0)

    def split_data(self, test_size, validation_size=None):
        """Splits all of the data into training/testing or training/testing/validation data

        Parameters
        ----------
        test_size : float
            A ratio of amount of testing data to all of the data.
        validation_size : float, optional
            A ratio of amount of validation data to all of the data.

        """

        if validation_size is None:
            self.X["train"], self.X["test"], self.y["train"], self.y["test"] = train_test_split(self.X["all"],
                                                                                                self.y["all"],
                                                                                                test_size=test_size)
        else:
            self.X["train"], X_remainder, self.y["train"], y_remainder = \
                train_test_split(self.X["all"], self.y["all"], test_size=test_size+validation_size)
            self.X["test"], self.X["validation"], self.y["test"], self.y["validation"] = \
                train_test_split(X_remainder, y_remainder, test_size=validation_size/(validation_size+test_size))

    @staticmethod
    def deconstruct_image(img, patch_shape):
        """Deconstructs a single image into several smaller patches.

        Parameters
        ----------
        img : ndarray
            Array of shape [height, width, n_channels].
        patch_shape : ndarray or tuple
            Array of shape [height, width, n_channels or n_classes]

        Returns
        -------
        patch : ndarray
            Array of shape [n_patches, height, width, n_channels or n_classes]

        """
        pad_y = img.shape[0] % patch_shape[0]
        pad_x = img.shape[1] % patch_shape[1]
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
                patch[i*n_x+j, :, :, 0] = img[y1:y2, x1:x2, 0]

        return patch

    @staticmethod
    def reconstruct_image(patch, image_shape):
        """Reconstructs several patches into a single image.

        Parameters
        ----------
        patch : ndarray
            Array of shape [n_patches, height, width, n_channels or n_classes].
        image_shape : ndarray or tuple
            Array of shape [height, width, n_channels or n_classes]

        Returns
        -------
        img : ndarray
            Array of shape [n_patches, height, width, n_channels or n_classes]

        """

        pad_y = image_shape[0] % patch.shape[1]
        pad_x = image_shape[1] % patch.shape[2]
        img = np.zeros((image_shape[0]+pad_y, image_shape[1]+pad_x, 1))

        n_y = int(img.shape[0] / patch.shape[1])
        n_x = int(img.shape[1] / patch.shape[2])

        for i in range(n_y):
            for j in range(n_x):
                x1 = j*patch.shape[2]
                x2 = (j+1)*patch.shape[2]
                y1 = i * patch.shape[1]
                y2 = (i + 1) * patch.shape[1]
                img[y1:y2, x1:x2, 0] = patch[i*n_x+j, :, :, 0]

        return img[:image_shape[0], :image_shape[1], 0]

    @staticmethod
    def extract_random_positive_patches(mask, patch_shape, num_patches, threshold=False, img=None):
        """Extracts patches from an image only at locations where mask is True.

        Parameters
        ---------
        img : ndarray
            Array to extract patches from of shape [height, width, n_channels].
        patch_shape : tuple
            Shape of the patches to be extracted of shape (height, width, n_channels).
        mask : ndarray
            Binary array of shape [height, width] where each pixel is represented by True or False.
        num_patches : int
            Number of positive patches to extract from the entire image.
        threshold : float
            A ratio of (True pixels / total pixels) that is used to determine if a patch is considered positive. If
            False, there only needs to be one true pixel.

        Returns
        -------
        patch : ndarray or list
            Array positive patches of shape [num_patches, patch_shape[0], patch_shape[1], patch_shape[2]] if
            'img' is provided as input. Otherwise, a list of bounding boxes is output where each item in the list is
            a list [x1, y1, x2, y2].

        """

        # Find indices where mask is True
        positive_idxs = np.transpose(mask.nonzero())

        # Decide on output
        if img is not None:
            patch = np.zeros((num_patches, patch_shape[0], patch_shape[1], img.shape[2], np.float32))
        else:
            patch = []

        # Loop until num_patches has been extracted
        extracted_patches = 0
        while extracted_patches != num_patches:

            valid_bbox = False
            while not valid_bbox:
                rand_idx = np.random.randint(0, positive_idxs.shape[0], 1)
                rand_pt = np.squeeze(positive_idxs[rand_idx])
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
        bbox = bbox_list[0]
        patch = np.zeros((len(bbox_list), bbox[3]-bbox[1], bbox[2]-bbox[0], img.shape[2]), img.dtype)
        for idx, bbox in enumerate(bbox_list):
            patch[idx] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        return patch


#######################################################################################################################
# Loss functions and metrics

smooth = 1.


# def binary_crossentropy(y_true, y_pred):
#     return -K.mean((y_true*K.log(y_pred)) + (1-y_true)*K.log(1-y_pred))


def precision_binary(y_true, y_pred):

    logical_and = K.cast(K.all(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    logical_or = K.cast(K.any(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    tp = K.sum(logical_and)
    fp = K.sum(logical_or - y_true)

    return tp / (tp + fp)


def recall_binary(y_true, y_pred):

    logical_and = K.cast(K.all(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    logical_or = K.cast(K.any(K.stack([K.cast(y_true, 'bool'), K.greater_equal(y_pred, 0.5)], axis=0), axis=0), 'float32')
    tp = K.sum(logical_and)
    fn = K.sum(logical_or - K.cast(K.greater_equal(y_pred, 0.5), 'float32'))

    return tp / (tp + fn)


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

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    return (intersection) / (sum_ - intersection)


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

    return 1 - jaccard(y_true, y_pred)


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
    return intersection / (sum_ - intersection)


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

    """
    intersection = K.sum(K.abs(K.flatten(y_true) * K.flatten(y_pred)))
    sum_ = K.sum(K.abs(K.flatten(y_true))) + K.sum(K.abs(K.flatten(y_pred)))
    return 2*(intersection+smooth) / (sum_+smooth)


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
    return 1 - dice(y_true, y_pred)


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
    return 2 * intersection / sum_


########################################################################################################################
# Callbacks

def schedule(epoch):
    """Custom Keras callback for creating a schedule for the learning rate based on the current epoch.

    Parameters
    ----------
    epoch : int
        The current epoch during training.

    Returns
    -------
    lr : float
        The learning rate to be used for training.

    """
    if epoch < 5:
        lr = .0001
    else:
        lr = .00001
    return lr

########################################################################################################################


if __name__ == "__main__":

    # Segmentation
    # y_gt = np.expand_dims(np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    #                                [[0, 1, 0], [1, 0, 1], [0, 1, 0]]], np.float64), axis=3)
    # y_out = np.expand_dims(np.array([[[0.9, 0.1, 0.9], [.9, 0.1, 0.75], [0.9, 0.8, 0.6]],
    #                                 [[0.2, 0.7, 0.3], [.9, 0.2, 0.95], [0.2, 0.8, 0.4]]]), axis=3)

    # Classification - binary
    y_gt = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    y_out = np.array([.9, .1, .2, .45, .3, .1, .1, .8, .6, .75, .4, .55])

    print("Precision = ", precision_binary(K.variable(y_gt), K.variable(y_out)).eval(session=K.get_session()))
    print("Recall = ", recall_binary(K.variable(y_gt), K.variable(y_out)).eval(session=K.get_session()))




