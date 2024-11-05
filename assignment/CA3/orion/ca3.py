################################# "FUNCTIONAL" INPUTS ################################# START
## Input
USER_NAME       = 'dat300-h24-30'
PATH_TO_DATASET_TRAIN = f'/mnt/users/{USER_NAME}/CA3/tree_train.h5'
PATH_TO_DATASET_TEST = f'/mnt/users/{USER_NAME}/CA3/tree_test.h5'

PATH_TO_STORE_PLOTS = f'/mnt/users/{USER_NAME}/CA3/bard'
################################# "FUNCTIONAL" INPUTS ################################# END

import os
import sys
import time
from tqdm import tqdm # Cool progress bar
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from tensorflow.keras.applications import EfficientNetB7
from sklearn.utils import shuffle

sys.path.append(os.path.abspath(".."))
from utilities import *

SEED = 458 # Feel free to set another seed if you want to
RNG = np.random.default_rng(SEED) # Random number generator
tf.random.set_seed(SEED)


with h5py.File(PATH_TO_DATASET_TRAIN,'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X'])
    y_train = np.asarray(f['y'])
    print('Nr. train images: %i'%(X_train.shape[0]))


random_numbers = RNG.integers(0, X_train.shape[0], size=5)
for i in random_numbers:
  image = X_train[i]
  mask = y_train[i]
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  axes[0].imshow(image)
  axes[0].set_title('Image')
  axes[1].imshow(mask, cmap='gray')
  axes[1].set_title('Mask')
  plt.savefig(f'{PATH_TO_STORE_PLOTS}/X_train_default.png')


# Turn to grey scale images
#X_train = np.expand_dims(X_train, -1)
X_train = X_train.astype("float32")/255

# Converting targets from numbers to categorical format
y_train = ks.utils.to_categorical(y_train, len(np.unique(y_train)))

zoom = tf.keras.layers.RandomZoom(0.4)
flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
rotate = tf.keras.layers.RandomRotation(0.6)
translation = tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
contrast = tf.keras.layers.RandomContrast(0.9)

def augment(combined_image):
    augmentations = [flip, translation, rotate] #, rotate, zoom, translation, contrast]
    num_augmentations = np.random.choice([1, 2])
    chosen_augmentations = np.random.choice(augmentations, num_augmentations, replace=False)
    for augmentation in chosen_augmentations:
        combined_image = augmentation(combined_image)

    rotated_x_image = combined_image[..., :3]
    rotated_y_image = combined_image[..., 3:]
    return rotated_x_image, rotated_y_image

def agument_all(X_train, y_train, n):
    augmented_X = []
    augmented_y = []

    for i in tqdm(range(n)):
        random_index = RNG.integers(0, X_train.shape[0])
        rotated_x_image, rotated_y_image = augment(np.concatenate([X_train[random_index], y_train[random_index]], axis=-1))
        augmented_X.append(rotated_x_image)
        augmented_y.append(rotated_y_image)

    # Convert lists to numpy arrays and concatenate with the original arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    X_train = np.concatenate([X_train, augmented_X], axis=0)
    y_train = np.concatenate([y_train, augmented_y], axis=0)

    return X_train, y_train

samples_to_generate = 1700
X_train, y_train = agument_all(X_train, y_train, samples_to_generate)

plt.figure(figsize=(12, 6))

for i in range(5):
    index = - (5 - i)
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[index])
    plt.axis("off")

    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(np.argmax(y_train[index], axis=-1), cmap='gray', alpha=0.5)
    plt.axis("off")

plt.tight_layout()
plt.savefig(f'{PATH_TO_STORE_PLOTS}/X_train_augmented.png')

# Shuffle X_train and y_train together
X_train, y_train = shuffle(X_train, y_train, random_state=42)

def double_conv_block(x, n_filters, kernel_size_=3, padding_="same", activation_ = "relu", batchnorm=True):
  x = ks.layers.Conv2D(filters = n_filters, kernel_size=[kernel_size_, kernel_size_],
                      padding = padding_, kernel_initializer = "he_normal")(x)
  if batchnorm:
    x = ks.layers.BatchNormalization()(x)

  x = ks.layers.Activation(activation_)(x)
  x = ks.layers.Conv2D(filters = n_filters, kernel_size=[kernel_size_, kernel_size_],
                      padding = padding_, kernel_initializer = "he_normal")(x)
  if batchnorm:
    x = ks.layers.BatchNormalization()(x)

  x = ks.layers.Activation(activation_)(x)
  return x

def downsample_block(x, n_filters, kernel_size_dcb=3, dropout=0.1, kernel_size_mp2d=2):
   f = double_conv_block(x, n_filters, kernel_size_dcb)
   p = ks.layers.MaxPool2D((kernel_size_mp2d,kernel_size_mp2d))(f)
   p = ks.layers.Dropout(dropout)(p)
   return f, p


def upsample_block(x, conv_features, n_filters, kernel_size_c2dt=3, stride_=2, dropout=0.1):
   x = ks.layers.Conv2DTranspose(n_filters, (kernel_size_c2dt, kernel_size_c2dt),
                                 strides=(stride_, stride_), padding="same")(x)
   x = ks.layers.concatenate([x, conv_features])
   x = ks.layers.Dropout(dropout)(x)
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model(image_size, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2):
  # inputs
  inputs = ks.layers.Input(image_size)

  # 1 - downsample
  c1, p1 = downsample_block(inputs, n_filters)
  # 2 - downsample
  c2, p2 = downsample_block(p1, n_filters*2)
  # 3 - downsample
  c3, p3 = downsample_block(p2, n_filters*4)
  # 4 - downsample
  c4, p4 = downsample_block(p3, n_filters*8)

  c5 = double_conv_block(p4, n_filters*16)

  # 6 - upsample
  u6 = upsample_block(c5, c4, n_filters*8)
  # 7 - upsample
  u7 = upsample_block(u6, c3, n_filters*4)
  # 8 - upsample
  u8 = upsample_block(u7, c2, n_filters*2)
  # 9 - upsample
  u9 = upsample_block(u8, c1, n_filters)

  # outputs
  outputs = ks.layers.Conv2D(n_classes, (1,1), padding="same", activation = "softmax")(u9)
  unet_model = ks.Model(inputs, outputs, name="U-Net")

  return unet_model

input_img = (128, 128, 3)
print(input_img)
model_unet = build_unet_model(input_img)
model_unet.summary()


early_stopping = ks.callbacks.EarlyStopping(monitor='val_F1_score',
                               patience=10,
                              mode='max',
                               restore_best_weights=True)

# Compile the model
model_unet.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=[FalseNegatives(),
                       FalsePositives(),
                       TrueNegatives(),
                       TruePositives(),
                       F1_score])

# Train the model
model_unet_history = model_unet.fit(X_train, y_train,
                             epochs=1000, #more on orion
                             batch_size=32,
                             validation_split=(1/8),
                             callbacks=[early_stopping])

model_unet.save(f'{PATH_TO_STORE_PLOTS}/model_unet.h5')

"""### Plotting performance"""

rawDF = pd.DataFrame(model_unet_history.history)
plotDF = pd.DataFrame()

try:
    # Find the number behind the _ in the keys
    number = rawDF.columns[1].split('_')[-1]
    rawDF = rawDF.rename(columns={'true_positives_' + number : 'true_positives', 'true_negatives_'+number: 'true_negatives', 'false_positives_'+number: 'false_positives', 'false_negatives_'+number: 'false_negatives',
                                    'val_true_positives_'+number: 'val_true_positives', 'val_true_negatives_'+number: 'val_true_negatives', 'val_false_positives_'+number: 'val_false_positives', 'val_false_negatives_'+number: 'val_false_negatives'})
except:
    pass

plotDF['Accuracy']     = (rawDF['true_positives'] + rawDF['true_negatives']) / (rawDF['true_positives'] + rawDF['true_negatives'] + rawDF['false_positives'] + rawDF['false_negatives'])
plotDF['val_Accuracy'] = (rawDF['val_true_positives'] + rawDF['val_true_negatives']) / (rawDF['val_true_positives'] + rawDF['val_true_negatives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

plotDF['IoU']          = rawDF['true_positives'] / (rawDF['true_positives'] + rawDF['false_positives'] + rawDF['false_negatives'])
plotDF['val_IoU']      = rawDF['val_true_positives'] / (rawDF['val_true_positives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

plotDF['F1_score']     = rawDF['F1_score']
plotDF['val_F1_score'] = rawDF['val_F1_score']

list_of_metrics=['Accuracy', 'F1_score', 'IoU']
train_keys = list_of_metrics
valid_keys = ['val_' + key for key in list_of_metrics]
nr_plots = len(list_of_metrics)
fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
for i in range(len(list_of_metrics)):
    ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
    ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
    ax[i].set_xlabel('Epoch')
    ax[i].set_title(list_of_metrics[i])
    ax[i].grid('on')
    ax[i].legend()
fig.tight_layout
plt.savefig(f'{PATH_TO_STORE_PLOTS}/model_unet_history.png')

print(model_unet_history.history['F1_score'][-1])

"""## Task 1.3 Visualize model predictions

Make a plot that illustrates the original image, the predicted mask, and the ground truth mask.

### Plotting
"""

# Predict masks for a few images in the training set
y_pred = model_unet.predict(X_train[:5])

# Convert predicted masks to binary
y_pred_binary = np.argmax(y_pred, axis=-1)

# Convert ground truth masks to binary
y_train_binary = np.argmax(y_train[:5], axis=-1)

# Plot the original image, predicted mask, and ground truth mask for each image
plt.figure(figsize=(15, 5))
for i in range(5):
  plt.subplot(3, 5, i + 1)
  plt.imshow(X_train[i].squeeze(), cmap='gray')

  plt.subplot(3, 5, i + 6)
  plt.imshow(y_pred_binary[i], cmap='gray')

  plt.subplot(3, 5, i + 11)
  plt.imshow(y_train_binary[i], cmap='gray')

plt.savefig(f'{PATH_TO_STORE_PLOTS}/model_unet_pred_{i}.png')

"""# Part 2: Implementing U-net with transfer learning

Implement a model with the U-net structure that you have learned about in the lectures, but now with a pre-trained backbone. There are many pre-trained back-bones to choose from. Pick freely from the selection here [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications), or here [Keras model scores](https://keras.io/api/applications/) (nicer table in the second link). Feel free to experiment with the number of layers, loss-function, batch-normalization, etc. Many of the backbones available are quite big, so you might find it quite time-consuming to train them on your personal computers. It might be expedient to only train them for 1-5 epochs on your PCs, and do the full training on Orion in Part 3.


#### For those with a dedicated graphics card (NVIDIA and AMD) Tensorflow or PyTorch (not syllabus)
And wants to experiment with their own compute resources (can be alot of fun)\
Tensorflow: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin \
PyTorch: https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows

## Task 2.1 Transfer learning model implementation

Implement a U-net model utilizing the pre-trained weights of a publically available network. **Remember to compile with the F1-score metric**.

### Transfer learning with U-net
"""

def unet_efficientnetb7(input_shape, dropout_rate=0.1):
    # Input layer
    inputs = ks.layers.Input(shape=input_shape)
    # Pretrained EfficientNetB7 encoder
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_tensor=inputs)
    # Encoder blocks (skip connections)
    block1 = base_model.get_layer('block1a_activation').output  # (None, 64, 64, 64)
    block2 = base_model.get_layer('block2a_activation').output  # (None, 32, 32, 192)
    block3 = base_model.get_layer('block3a_activation').output  # (None, 16, 16, 288)
    block4 = base_model.get_layer('block5a_activation').output  # (None, 8, 8, 960)

    # Middle block
    block5 = base_model.get_layer('top_activation').output      # (None, 4, 4, 256)

    # Decoder blocks
    x5 = upsample_block(block5, block4, 960) # (None, 8, 8, 960)
    x4 = upsample_block(x5, block3, 288)    # (None, 16, 16, 288)
    x3 = upsample_block(x4, block2, 192)   # (None, 32, 32, 192)
    x2 = upsample_block(x3, block1, 64)  # (None, 64, 64, 64)

    # Final upsample and output layer
    x = ks.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x2)  # Final upsample layer (None, 128, 128, 32)
    output = ks.layers.Conv2D(2, (1, 1), padding="same", activation="softmax")(x)  # Output layer (None, 128, 128, 1)

    # Create the model
    unet_model = ks.models.Model(inputs, output)

    return unet_model

input_img = (128, 128, 3)
model_pre = unet_efficientnetb7(input_img)
model_pre.summary()

"""## Task 2.2 Train the transfer learning model and plot the training history

Feel free to use the `plot_training_history` function from the provided library `utilities.py`

### Training
"""

early_stopping = ks.callbacks.EarlyStopping(monitor='val_F1_score',
                               patience=10,
                              mode='max',
                               restore_best_weights=True)

# Compile the model
model_pre.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=[FalseNegatives(),
                       FalsePositives(),
                       TrueNegatives(),
                       TruePositives(),
                       F1_score])

# Train the model
model_pre_history = model_pre.fit(X_train, y_train,
                             epochs=1000, # more on orion
                             batch_size=32,
                             validation_split=(1/8),
                             callbacks=[early_stopping])

model_pre.save(f'{PATH_TO_STORE_PLOTS}/model_pre.h5')


"""### Plotting"""

rawDF = pd.DataFrame(model_pre_history.history)
plotDF = pd.DataFrame()

try:
    # Find the number behind the _ in the keys
    number = rawDF.columns[1].split('_')[-1]
    rawDF = rawDF.rename(columns={'true_positives_' + number : 'true_positives', 'true_negatives_'+number: 'true_negatives', 'false_positives_'+number: 'false_positives', 'false_negatives_'+number: 'false_negatives',
                                    'val_true_positives_'+number: 'val_true_positives', 'val_true_negatives_'+number: 'val_true_negatives', 'val_false_positives_'+number: 'val_false_positives', 'val_false_negatives_'+number: 'val_false_negatives'})
except:
    pass

plotDF['Accuracy']     = (rawDF['true_positives'] + rawDF['true_negatives']) / (rawDF['true_positives'] + rawDF['true_negatives'] + rawDF['false_positives'] + rawDF['false_negatives'])
plotDF['val_Accuracy'] = (rawDF['val_true_positives'] + rawDF['val_true_negatives']) / (rawDF['val_true_positives'] + rawDF['val_true_negatives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

plotDF['IoU']          = rawDF['true_positives'] / (rawDF['true_positives'] + rawDF['false_positives'] + rawDF['false_negatives'])
plotDF['val_IoU']      = rawDF['val_true_positives'] / (rawDF['val_true_positives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

plotDF['F1_score']     = rawDF['F1_score']
plotDF['val_F1_score'] = rawDF['val_F1_score']

list_of_metrics=['Accuracy', 'F1_score', 'IoU']
train_keys = list_of_metrics
valid_keys = ['val_' + key for key in list_of_metrics]
nr_plots = len(list_of_metrics)
fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
for i in range(len(list_of_metrics)):
    ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
    ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
    ax[i].set_xlabel('Epoch')
    ax[i].set_title(list_of_metrics[i])
    ax[i].grid('on')
    ax[i].legend()
fig.tight_layout
plt.savefig(f'{PATH_TO_STORE_PLOTS}/model_pre_history.png')

"""# Part 3: Training your model Orion

Use the lecture slides from the Orion-lecture to get started.
1. Put one of your model implementations into a python script (`.py`)
2. Transfer that script to Orion.
3. Change the relevant path variables in your python script (path-to-data for example), and make sure that you record the time it takes to train the model in the script. This can be done using the `time` library for example.
4. Set up a SLURM-script to train your model, please use the example from the Orion lecture as a base.
5. Submit your SLURM job, and let the magic happen.

If you wish to use a model trained on Orion to make a Kaggle submission, remember to save the model, such that you can transfer it to your local computer to make a prediction on `X_test`, or test the model on Orion directly if you want to.

## Tips

If you compiled, trained and stored a model on Orion with a custom performance metric (such as F1-score), remember to specify that metric when loading the model on your computer again.

Loading a saved model:
```python
trained_model = tf.keras.models.load_model('some/path/to/my_trained_model.keras', custom_objects={'F1_score': F1_score})
```

Loading a checkpoint:
```python
trained_model = tf.keras.saving.load_model('some/path/to/my_trained_model_checkpoint', custom_objects={'F1_score': F1_score})
```

# Discussion

**Question 1: Which model architectures did you explore, and what type of hyperparameter optimization did you try?**

**Answer 1:**

**Question 2: Which of the model(s) did you choose to train on Orion, and how long did it take to train it on Orion?**

**Answer 2:**

**Question 3: What where the biggest challenges with this assignment?**

**Answer 3:**

# Kaggle submission

Evaluate your best model on the test dataset and submit your prediction to the Kaggle leaderboard.
Link to the Kaggle leaderboard will be posted in the Canvas assignment.

### Kaggle submission
"""

with h5py.File(PATH_TO_DATASET_TEST,'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_test = np.asarray(f['X'])
    print('Nr. train images: %i'%(X_test.shape[0]))

# Turn to grey scale images
X_test = X_test.astype("float32")/255

# Threshold for sigmoid
USER_DETERMINED_THRESHOLD = 0.5

y_pred      = model_unet.predict(X_test)  # Make prediction using the trained model
flat_y_pred = y_pred.flatten()                             # Flatten prediction
flat_y_pred[flat_y_pred >= USER_DETERMINED_THRESHOLD] = 1  # Binarize prediction (Optional, depends on output activation used)
flat_y_pred[flat_y_pred != 1]   = 0                        # Binarize prediction (Optional, depends on output activation used)
submissionDF = pd.DataFrame()
submissionDF['ID'] = range(len(flat_y_pred))               # The submission csv file must have a column called 'ID'
submissionDF['Prediction'] = flat_y_pred
submissionDF.to_csv(f'{PATH_TO_STORE_PLOTS}/submission.csv', index=False)

"""### Kaggle submission"""
# Make shure predictions look ok
plt.figure(figsize=(12, 6))
for i in range(6):
    image_x = np.squeeze(X_test[i])
    plt.subplot(2, 6, i + 1)
    plt.imshow(image_x)
    plt.axis("off")
    plt.subplot(2, 6, i + 7)
    plt.imshow(np.argmax(y_pred[i], axis=-1), cmap='gray', alpha=0.5)
    plt.axis("off")
plt.savefig(f'{PATH_TO_STORE_PLOTS}/X_test_predictions.png')
