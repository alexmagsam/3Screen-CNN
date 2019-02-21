import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from sklearn.metrics import *
from .networks import *
from .utils import *


class Model:
    """
    A parent class used for training a CNN for image classification or semantic segmentation.

    ...

    Attributes
    ----------
    model: keras Model
        Contains the architecture of the CNN, the trained weights, and the attributes and methods of a keras Model.

    Methods
    -------
    build_model(config)
        Builds the desired CNN as a keras Model specified by the config parameter.
    train(dataset, config)
        Used to build the keras Model and train the weights of the CNN on the dataset based on parameters specified
        by the config parameter.
    evaluate_test(X_test, y_test, batch_size, to_csv=False, save_path='')
        Evaluates the trained network on the test data using classification metrics or segmentation metrics depending on
        the output shape of the predicted data.
    visualize_patch_segmentation_predictions(X, y=None, num_predictions=3)
        Displays a number of random patch segmentation masks predicted by the model attribute of the class.
    visualize_full_segmentation_predictions(img, mask=None, threshold=.5)
        Displays an entire segmentation mask of an image larger than the input shape of the trained CNN by splitting
        into patches, predicting the segmentation masks of the patches, and then reassembling the patches to the
        original shape of the image.

    """

    model = None

    def __init__(self):
        pass

    def build_model(self, config):
        """Sets the model attribute of the class by creating the keras Model according to the config parameter.

        Parameters
        ----------
        config : Config object
            This class contains the attribute MODEL_NAME that specifies which CNN model to use for training.

        Returns
        -------
        model.summary() : str
            A summary of the CNN architecture layer by layer.

        """

        if config.MODEL_NAME.lower() == 'u-net-small':
            self.model = unet_small(config.INPUT_SHAPE, config.NUM_CLASSES)
        elif config.MODEL_NAME.lower() == 'u-net':
            self.model = unet(config.INPUT_SHAPE, config.NUM_CLASSES)
        elif config.MODEL_NAME.lower() == 'u-net-original':
            self.model = unet_original(config.NUM_CLASSES)
        elif config.MODEL_NAME.lower() == 'inceptionv3':
            self.model = inceptionv3(config.INPUT_SHAPE, config.NUM_CLASSES, config.WEIGHTS)
        elif config.MODEL_NAME.lower() == 'vgg16':
            self.model = vgg16(config.INPUT_SHAPE, config.NUM_CLASSES, config.WEIGHTS)
        else:
            raise ValueError("Choose a valid model name.")

        # Choose the loss function
        if config.LOSS.lower() == 'bce':
            loss = 'binary_crossentropy'
        elif config.LOSS.lower() == 'cce':
            loss = 'categorical_crossentropy'
        elif config.LOSS.lower() == 'jaccard':
            loss = jaccard_loss
        elif config.LOSS.lower() == 'dice':
            loss = dice_loss
        else:
            raise ValueError("Select a valid loss function")

        # Choose the optimizer
        if config.OPTIMIZER["name"].lower() == 'adam':
            optimizer = Adam(config.LEARNING_RATE, decay=config.OPTIMIZER["decay"])
        elif config.OPTIMIZER["name"].lower() == 'sgd':
            optimizer = SGD(config.LEARNING_RATE, momentum=config.OPTIMIZER["momentum"],
                            decay=config.OPTIMIZER["decay"])
        elif config.OPTIMIZER["name"].lower() == 'rmsprop':
            optimizer = RMSprop(config.LEARNING_RATE, epsilon=config.OPTIMIZER["epsilon"],
                                decay=config.OPTIMIZER["decay"])
        else:
            raise ValueError("Select a valid optimizer")

        # Choose the appropriate metrics
        if config.MODEL_NAME.lower() in ["u-net", "u-net-small"]:
            metrics = [dice, jaccard, K.binary_crossentropy]
        elif config.NUM_CLASSES == 1:
            metrics = ['accuracy', precision_binary, recall_binary]
        else:
            metrics = ['accuracy']

        # Compile the model
        self.model.compile(optimizer, loss=[loss], metrics=metrics)

        return self.model.summary()

    def train(self, dataset, config, train_generator=None, val_generator=None):
        """Trains the CNN model attribute of the class on the dataset according to the configuration parameter.

        Parameters
        ---------
        dataset : Dataset object
            Provides the training, testing, and validation data for training the CNN.
        config : Config object
            Specifies the hyper-parameters of CNN for training including loss function and optimizer specifications.
        generator : ImageGenerator object
        val_generator : ImageGenerator object

        """

        if self.model is None:
            self.build_model(config)
        else:
            print("Model already created")

        # Create training directories
        config.create_training_directories()

        # Create a save path for training
        callbacks = []

        # Set a  callback for the model to save checkpoints
        filename = 'model.{epoch:02d}-{val_loss:.2f}.hdf5'
        callbacks.append(ModelCheckpoint(os.path.join(config.MODELS_DIR, filename), save_best_only=True))

        # Set a callback to log training values to a CSV
        callbacks.append(CSVLogger(os.path.join(config.CSV_DIR, 'training.csv')))

        # Set a callback to adjust the learning rate
        if config.LR_SCHEDULER:
            callbacks.append(LearningRateScheduler(schedule))

        # Train the model
        if train_generator is None and val_generator is None:
            self.model.fit(dataset.X["train"], dataset.y["train"], epochs=config.NUM_EPOCHS, callbacks=callbacks,
                           validation_data=(dataset.X["validation"], dataset.y["validation"]),
                           batch_size=config.BATCH_SIZE)
        else:
            self.model.fit_generator(train_generator, steps_per_epoch=train_generator.n // train_generator.batch_size,
                                     validation_data=val_generator,
                                     validation_steps=val_generator.n // val_generator.batch_size,
                                     epochs=config.NUM_EPOCHS, callbacks=callbacks)

    def evaluate_test(self, dataset, config, to_csv=False):
        """Measures the performance of the trained CNN on test data.

        Parameters
        ----------
        dataset : Dataset Object
        config : Config Object
            Provides parameters for inference
        to_csv : bool, optional
            If True, a csv file will be saved with the calculated metrics.

        Returns
        -------
        df : Dataframe Object
            A dataframe containing all of the relevant information from the training session.

        """

        if dataset.X["test"].any() and dataset.y["test"].any():
            X_test = dataset.X["test"]
            y_test = dataset.y["test"]
        else:
            X_test = dataset.X["validation"]
            y_test = dataset.y["validation"]

        y_pred = self.model.predict(X_test, config.BATCH_SIZE)
        if y_test.ndim == 4:    # Segmentation
            score = {"BCE": binary_crossentropy_np(y_test, y_pred),
                     "Dice": dice_np(y_test, y_pred),
                     "Jaccard": jaccard_np(y_test, y_pred)}
        elif y_test.ndim == 1:     # Binary classification
            y_pred = np.squeeze(y_pred)
            score = {"Accuracy": accuracy_score(y_test, y_pred >= 0.5),
                     "Precision": precision_score(y_test, y_pred >= 0.5),
                     "Recall": recall_score(y_test, y_pred >= 0.5),
                     "F1": f1_score(y_test, y_pred >= 0.5),
                     "AUC": roc_auc_score(y_test, y_pred),
                     "AP": average_precision_score(y_test, y_pred)}
        elif y_test.ndim == 2:     # Multi-class classification
            class_predictions = np.argmax(y_pred, axis=1)
            test_predictions = np.argmax(y_test, axis=1)
            score = {"Accuracy": accuracy_score(test_predictions, class_predictions),
                     "Average Precision": average_precision_score(y_test, y_pred),
                     "AUC": roc_auc_score(y_test, y_pred)}
        else:
            raise ValueError("Output format not recognized")

        # Create a dictionary from the dataset and config and update it with the score dictionary
        summary = training_summary_dict(dataset, config)
        summary.update(score)
        df = pd.DataFrame(summary, index=['value'])

        if to_csv:
            df.to_csv(os.path.join(config.CSV_DIR, 'results.csv'))

        return df

    def evaluate_test_generator(self, dataset, config, to_csv=False):
        # TODO: Create this method
        pass

    def visualize_class_predictions(self, X, y=None, class_names=None, threshold=.5, num_predictions=3):
        """Visualize randomly selected predictions for a CNN trained for classification.

        Parameters
        ----------
        X : ndarray
            An array with shape [n_samples, height, width, n_channels].
        y : ndarray, optional
            Ground truth array with shape [n_samples] or [n_samples, n_classes].
        class_names : list
            A list containing the names of the classes. The index of the class name corresponds to the class value.
        threshold : float, optional
            Threshold to use to convert predicted output to binary output.
        num_predictions : int, optional
            The number of random predictions to display.

        """

        random_samples = np.random.randint(0, len(X), num_predictions)

        X_rand = X[random_samples]
        y_pred = self.model.predict(X_rand)
        y_pred = np.squeeze(y_pred)

        if y_pred.ndim == 1:
            class_predictions = (y_pred >= threshold).astype('int')
        elif y_pred.ndim == 2:
            class_predictions = np.argmax(y_pred, axis=1).astype('int')
        else:
            ValueError("Accepted dimensions are (n_samples, ) or (n_samples, n_classes).")

        # TODO: only ten classes
        if class_names is None:
            class_names = [i for i in range(10)]

        if y is not None:
            y_rand = y[random_samples]
            if y_rand.ndim == 1:
                true_class = (y_rand >= threshold).astype('int')
            elif y_rand.ndim == 2:
                true_class = np.argmax(y_rand, axis=1).astype('int')
            else:
                ValueError("Accepted dimensions are (n_samples, ) or (n_samples, n_classes).")

        ncols = min(num_predictions, 5)
        nrows = 1
        fig, axes = plt.subplots(nrows, ncols)

        for idx in range(ncols):
            axes[idx].imshow(np.squeeze(X_rand[idx]))
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

            if y_pred.ndim == 2:
                probability = 100*y_pred[idx, class_predictions[idx]]
            else:
                probability = 100*y_pred[idx] if class_predictions[idx] == 1 else 100*(1-y_pred[idx])

            if y is not None:
                title = "Predicted: {:.2f}% {}, True: {}".format(probability,
                                                                 class_names[class_predictions[idx]],
                                                                 class_names[true_class[idx]])
            else:
                title = "Predicted: {:.2f}% {}".format(probability,
                                                       class_names[class_predictions[idx]])

            axes[idx].set_title(title, fontsize=8)

        plt.show()

    def visualize_patch_segmentation_predictions(self, X, y=None, threshold=0.5, num_predictions=3):
        """Visualize randomly selected predictions for a CNN trained for semantic segmentation.

        Parameters
        ----------
        X : ndarray
            An array with shape [n_samples, height, width, n_channels].
        y : ndarray, optional
            An array with the same shape as X used as the ground truth segmentation.
        threshold : float, optional
            Threshold to use to convert predicted output to binary output.
        num_predictions : int, optional
            The number of random predictions to display.

        """

        random_samples = np.random.randint(0, len(X), num_predictions)

        X_rand = X[random_samples]
        y_pred = self.model.predict(X_rand)

        ncols = 2
        nrows = num_predictions
        if y is not None:
            ncols = 3
            y_rand = y[random_samples]

        fig, axes = plt.subplots(nrows, ncols)

        for idx in range(num_predictions):
            axes[idx, 0].imshow(X_rand[idx, :, :, 0], cmap='gray')
            axes[idx, 0].set_xticks([])
            axes[idx, 0].set_yticks([])

            axes[idx, 1].imshow(y_pred[idx, :, :, 0] > threshold, cmap='gray')
            axes[idx, 1].set_xticks([])
            axes[idx, 1].set_yticks([])

            if idx == 0:
                axes[idx, 0].set_title("Original Image")
                axes[idx, 1].set_title("Predicted Mask")

            if y is not None:
                axes[idx, 2].imshow(y_rand[idx, :, :, 0], cmap='gray')
                axes[idx, 2].set_xticks([])
                axes[idx, 2].set_yticks([])
                if idx == 0:
                    axes[idx, 2].set_title("Ground Truth Mask")

        plt.show()

    def visualize_full_segmentation_predictions(self, img, mask=None, threshold=.5):
        """Visualize the segmentation of a single image larger than the input size of the CNN.

        Parameters
        ----------
        img : ndarray
            An array with shape [height, width, channels] to perform the segmentation on.
        mask : ndarray, optional
            Ground truth mask to compare the predicted segmentation to.
        threshold : float, optional
            Threshold to use to convert predicted output to binary output.
        """

        X = Dataset.deconstruct_image(img, self.model.input_shape[1:])
        y_pred = self.model.predict(X)

        mask_pred = Dataset.reconstruct_image(y_pred, img.shape)

        ncols = 2
        nrows = 1
        if mask is not None:
            ncols = 3

        fig, axes = plt.subplots(nrows, ncols)

        if img.shape[2] == 1:
            axes[0].imshow(img[..., 0], cmap='gray')
        else:
            axes[0].imshow(img)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title("Image")

        if mask_pred.shape[2] == 1:
            axes[1].imshow(np.squeeze(mask_pred >= threshold), cmap='gray')
        else:
            axes[1].imshow(np.argmax(mask_pred, axis=2), cmap='jet')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title("Predicted Mask")

        if mask is not None:
            axes[1].set_title("Predicted - IoU = {0:.2f}".format(jaccard_np(mask, mask_pred)))
            if mask.shape[2] == 1:
                axes[2].imshow(mask[..., 0], cmap='gray')
            else:
                axes[2].imshow(np.argmax(mask, axis=2), cmap='jet')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            axes[2].set_title("Ground Truth")
        plt.show()







