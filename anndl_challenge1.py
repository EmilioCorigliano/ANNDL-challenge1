"""ANNDL-challenge1.ipynb

Python script for the creation and training of a CNN for the resolution of Challenge 1 of ANNDL course

Original file is located at
    https://colab.research.google.com/drive/1TK6Ml5cFjPaJaMxQKY_hncjzBwp5uSph
"""

from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

tfk = tf.keras
tfkl = tf.keras.layers

"""
!pip install visualkeras
!pip install scikit-learn
!pip install scikit-image
!pip install pyyaml
!pip install imutils
!pip install opencv-python
!pip install tensorboard
!pip install tensorflow_addons
!pip install google.colab
"""

print(tf.__version__)

# use GPUS in local computer
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

## """PARAMETERS"""

# path definitions
dataset_dir = "C:\\Users\\emili\\OneDrive - Politecnico di " \
              "Milano\\Desktop\\Backup\\POLITECNICO\\5ANNO\\1-ANNDL\\laboratory\\ANNDL-challenge1"
training_dir = os.path.join(dataset_dir, 'training_data_final')
# if path_tl=="" then we perform the transfer learning part, otherwise we use the path_tl to load that
path_tl = os.path.join(dataset_dir, "data_augmentation_tl_challenge_1\\CNN_Aug_tl_Best_Nov21_22-51-21")
# if path_ft=="" then we perform the fine tuning part, otherwise we use the path_ft to load that
path_ft = os.path.join(dataset_dir, "data_augmentation_tl_challenge_1\\CNN_Aug_ft_Best_Nov24_11-10-41")

# preprocessing
preprocessing_function = tf.keras.applications.vgg19.preprocess_input
preprocessing_function_name = "vgg19"

# other parameters
seed = 42
batch_size = 8

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# define labels
labels = ['species 1',  # 0
          'species 2',  # 1
          'species 3',  # 2
          'species 4',  # 3
          'species 5',  # 4
          'species 6',  # 5
          'species 7',  # 6
          'species 8']  # 7

# other settings
font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 22}
# mpl.use('Qt5Agg')
mpl.rc('font', **font)


## """VISUALIZE BATCH"""


def get_next_batch(generator):
    """
    Function to take the batch and plot his first image with relative informations
    :param generator: batches iterator
    :return: gives the next batch from the DirectoryIterator
    """
    batch = next(generator)

    image = batch[0]  # first position is the image
    target = batch[1]  # second position is the target

    print("(Input) image shape:", image.shape)
    print("Target shape:", target.shape)

    # Visualize only the first sample
    image = image[0]
    target = target[0]
    target_idx = np.argmax(target)
    print()
    print("Categorical label:", target)
    print("Label:", target_idx)
    print("Class name:", labels[target_idx])
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(np.uint8(image))

    return batch


def create_folders_and_callbacks(model_name, target_dir, patience):
    """
    Function that creates the folder in the <target_dir> for a model called <model_name> and creates the function
    callbacks for checkpoint generation, visualization learning on Tensorboard and Early Stopping
    :param model_name: the name of the directory where all the infos will be stored
    :param target_dir: the path of the directory where to put the directory of this model
    :param patience: the patience parameter for this model
    :return: the array of callbacks generated
    """

    exps_dir = os.path.join(target_dir, 'data_augmentation_tl_challenge_1')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    # ----------------
    ckpt_dir = os.path.join(exp_dir, 'ckpts_challenge_1')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'),
                                                       save_weights_only=False,  # True to save only weights
                                                       save_best_only=False)  # True to save only the best epoch
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    # ---------------------------------
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if > 0 (epochs) shows weights histograms
    callbacks.append(tb_callback)

    # Early Stopping
    # --------------
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max',
                                                   restore_best_weights=True)
    callbacks.append(es_callback)

    return callbacks


def plot_dataset():
    """
    Generation of the dataset without preprocessing in order to plot images during execution of the script
    :return: a batches iterator for the images to plot
    """
    plot_data_generator = ImageDataGenerator()
    return plot_data_generator.flow_from_directory(directory=training_dir, target_size=(96, 96),
                                                   color_mode='rgb',
                                                   classes=None, batch_size=batch_size, shuffle=False, seed=seed,
                                                   subset='training')


plot_data_gen = plot_dataset()

# Get a sample from dataset and show info
_ = get_next_batch(plot_data_gen)


## """NO AUGMENTATION"""

def no_augmentation():
    """
    Functions that generates the batches iterator for the training and the validation phases without augmentation
    :return: array composed of two elements, training batches iterator and validation batches iterator
    """
    noaug_train_data_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2,
                                              preprocessing_function=preprocessing_function)
    noaug_train_gen_loc = noaug_train_data_gen.flow_from_directory(directory=training_dir, target_size=(96, 96),
                                                                   color_mode='rgb',
                                                                   classes=None, batch_size=batch_size, shuffle=True,
                                                                   seed=seed,
                                                                   subset='training')
    noaug_valid_gen_loc = noaug_train_data_gen.flow_from_directory(directory=training_dir, target_size=(96, 96),
                                                                   color_mode='rgb',
                                                                   classes=None, batch_size=batch_size, shuffle=False,
                                                                   seed=seed,
                                                                   subset='validation')

    # check classes
    print('Assigned labels')
    print(noaug_train_gen_loc.class_indices)
    print()
    print('Target classes')
    print(noaug_train_gen_loc.classes)

    return [noaug_train_gen_loc, noaug_valid_gen_loc]


[noaug_train_gen, noaug_valid_gen] = no_augmentation()


## """AUGMENTATION"""

def augmentation():
    """
    Function that generates the batches iterator for the augmented data. This also plots all the
    modifications applied and, at the end, the difference between the original and the augmented one.
    :return: batches iterator for augmented data
    """
    aug_train_data_gen = ImageDataGenerator(rotation_range=30,
                                            height_shift_range=50,
                                            width_shift_range=50,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='reflect',
                                            rescale=1 / 255.,
                                            preprocessing_function=preprocessing_function)

    aug_train_gen = aug_train_data_gen.flow_from_directory(directory=training_dir, target_size=(96, 96),
                                                           color_mode='rgb',
                                                           classes=None, batch_size=batch_size, shuffle=True, seed=seed)

    # Get sample image
    image = next(plot_data_gen)[0][4]

    # Create an instance of ImageDataGenerator for each transformation
    rot_gen = ImageDataGenerator(rotation_range=30)  # rotated randomly of +/- 30 deg
    shift_gen = ImageDataGenerator(width_shift_range=50)  # shift randomly of a value ranging from -50 to 50 pixels
    zoom_gen = ImageDataGenerator(zoom_range=0.3)  # maximum 30% zoomed
    flip_gen = ImageDataGenerator(horizontal_flip=True)  # flip horizontally

    # Get random transformations
    rot_t = rot_gen.get_random_transform(img_shape=(256, 256), seed=seed)
    print('Rotation:', rot_t, '\n')
    shift_t = shift_gen.get_random_transform(img_shape=(256, 256), seed=seed)
    print('Shift:', shift_t, '\n')
    zoom_t = zoom_gen.get_random_transform(img_shape=(256, 256), seed=seed)
    print('Zoom:', zoom_t, '\n')
    flip_t = flip_gen.get_random_transform(img_shape=(256, 256), seed=seed)
    print('Flip:', flip_t, '\n')

    # Apply the transformation
    gen = ImageDataGenerator(fill_mode='constant', cval=0.)
    rotated = gen.apply_transform(image, rot_t)
    shifted = gen.apply_transform(image, shift_t)
    zoomed = gen.apply_transform(image, zoom_t)
    flipped = gen.apply_transform(image, flip_t)

    # Plot original and augmented images
    fig, ax = plt.subplots(1, 5, figsize=(25, 10))
    ax[0].imshow(np.uint8(image))
    ax[0].set_title('Original')
    ax[1].imshow(np.uint8(rotated))
    ax[1].set_title('Rotated')
    ax[2].imshow(np.uint8(shifted))
    ax[2].set_title('Shifted')
    ax[3].imshow(np.uint8(zoomed))
    ax[3].set_title('Zoomed')
    ax[4].imshow(np.uint8(flipped))
    ax[4].set_title('Flipped')

    # Combine multiple transformations
    gen = ImageDataGenerator(rotation_range=30,
                             height_shift_range=50,
                             width_shift_range=50,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

    # Get random transformation
    t = gen.get_random_transform(img_shape=(256, 256), seed=seed)
    print("Transform:", t)

    # Apply the transformation
    augmented = gen.apply_transform(image, t)

    # Plot original and augmented images
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    ax[0].imshow(np.uint8(image))
    ax[0].set_title("Original")
    ax[1].imshow(np.uint8(augmented))
    ax[1].set_title("Augmented")
    plt.show()

    return aug_train_gen


aug_train_gen = augmentation()


## """TRANSFER LEARNING"""

def transfer_learning_vgg19():
    """
    Applies the transfer learning from the deep CNN VGG19. Here we use VGG19 as the feature extractor and we add layers
    for the recognizer. After the training we save our model and we also plot some information about the accuracy and
    crossentropy trends of our training.
    :return: directory path of our trained model
    """
    print("*** TRANSFER LEARNING ***")

    # download and plot preprocessing_function_name model
    vgg19 = tfk.applications.vgg19.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3)
    )
    vgg19.summary()

    # Use vgg19 as feature extractor
    vgg19.trainable = False

    # Add dense top
    inputs = tfk.Input(shape=(96, 96, 3))
    x = tfkl.Resizing(96, 96, interpolation="bicubic")(inputs)
    x = vgg19(x)
    x = tfkl.AveragePooling2D(pool_size=(2, 2), name='Pooling-2')(x)
    x = tfkl.Flatten(name='Flattening')(x)
    x = tfkl.Dropout(0.5, seed=seed)(x)
    x = tfkl.Dense(
        1024,
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)
    x = tfkl.Dropout(0.5, seed=seed)(x)
    outputs = tfkl.Dense(
        8,
        activation='softmax',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)

    # Create folders and callbacks
    aug_tl_callbacks = create_folders_and_callbacks(model_name='CNN_Aug_tl', target_dir=dataset_dir, patience=20)

    # Connect input and output through the Model class
    aug_tl_model = tfk.Model(inputs=inputs, outputs=outputs, name="vgg19")

    # Compile the model
    aug_tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')
    aug_tl_model.summary()

    # Train the model
    input_shape = (96, 96, 3)
    epochs = 200

    aug_tl_history = aug_tl_model.fit(
        x=aug_train_gen,
        epochs=epochs,
        validation_data=noaug_valid_gen,
        callbacks=aug_tl_callbacks,
    ).history

    # Save best epoch model
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    filename = dataset_dir + "/data_augmentation_challenge_1/CNN_Aug_tl_vgg19_Best_" + str(now)
    aug_tl_model.save(filename)
    del aug_tl_model

    # Plot the training
    plt.figure(figsize=(15, 5))
    plt.plot(aug_tl_history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(aug_tl_history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(aug_tl_history['accuracy'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(aug_tl_history['val_accuracy'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()
    return filename


if not path_tl:
    # If the path_tl is empty do the transfer learning phase, otherwise go directly to ft
    path_tl = transfer_learning_vgg19()
    print(path_tl)


## """FINE TUNING"""

def fine_tuning(path):
    """
    Finally we apply fine tuning on the model saved in the path passed as parameter. After the training we save our
    model and we also plot some information about the accuracy and crossentropy trends of our training.
    :param path: The path of the model on which we will apply fine tuning
    :return: directory path of our trained model
    """
    print("*** FINE TUNING ***")
    epochs = 1000

    # Re-load the model after transfer learning
    aug_ft_model = tfk.models.load_model(path)
    aug_ft_model.summary()

    # Set all vgg19 layers to True
    aug_ft_model.get_layer('vgg19').trainable = True
    for i, layer in enumerate(aug_ft_model.get_layer('vgg19').layers):
        print(i, layer.name, layer.trainable)

    # Freeze first N layers
    for i, layer in enumerate(aug_ft_model.get_layer('vgg19').layers[:0]):
        layer.trainable = False
    for i, layer in enumerate(aug_ft_model.get_layer('vgg19').layers):
        print(i, layer.name, layer.trainable)
    aug_ft_model.summary()

    # Compile the model
    aug_ft_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(1e-4),
                         metrics='accuracy')

    # Create folders and callbacks and fit
    aug_ft_callbacks = create_folders_and_callbacks(model_name='CNN_Aug_ft', target_dir=dataset_dir, patience=50)

    # Fine-tune the model
    aug_ft_history = aug_ft_model.fit(
        x=aug_train_gen,
        epochs=epochs,
        validation_data=noaug_valid_gen,
        callbacks=aug_ft_callbacks,
    ).history

    # Plot the training
    plt.figure(figsize=(15, 5))
    plt.plot(aug_ft_history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(aug_ft_history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(aug_ft_history['accuracy'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(aug_ft_history['val_accuracy'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()

    # Save best epoch model
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    save_path = dataset_dir + "/data_augmentation_challenge_1/CNN_Aug_ft_Best_" + str(now)
    aug_ft_model.save(save_path)
    return save_path


if not path_ft:
    # If the path_ft is empty do the fine tuning phase, otherwise go directly to the outcomes
    path_ft = fine_tuning(path_tl)
    print(path_ft)


## """CONFUSION MATRIX"""


def confusion_matrix_plot(path, dataset):
    """
    Function to plot the confusion matrix
    :param path: The path of the model to analyze
    :param dataset: The validation dataset
    """
    print("*** CONFUSION MATRIX ***")

    model = tfk.models.load_model(path)

    # Confution Matrix and Classification Report
    Y_pred = model.predict(dataset, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=-1)
    print('Confusion Matrix')
    cm = confusion_matrix(dataset.classes, y_pred)
    print('Classification Report')
    target_names = labels
    print(classification_report(dataset.classes, y_pred, target_names=target_names))

    # Plot the confusion matrix
    plt.figure(figsize=(30, 25))
    sns.heatmap(cm.T, xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()


confusion_matrix_plot(path_tl, noaug_valid_gen)
confusion_matrix_plot(path_ft, noaug_valid_gen)

# tensorboard
# tensorboard --logdir /Users/aless/PycharmProjects/pythonProject/ANN2DL-Challenge_1/Database/data_augmentation_tl_challenge_1
