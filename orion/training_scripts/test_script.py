################################# "FUNCTIONAL" INPUTS ################################# START
## Input
USER_NAME       = 'dat300-h24-50'
PATH_TO_DATASET = f'/mnt/users/{USER_NAME}/student_dataset_CIFAR10.h5'
MODEL_NAME      = 'test_model_dat300'

# Parameters
BATCH_SIZE = 128
EPOCHS = 20

## Outputs
# Model file
PATH_TO_STORE_MODEL = f'/mnt/users/{USER_NAME}/models'
################################# "FUNCTIONAL" INPUTS ################################# END

# Imports
import os
from time import time
from tqdm import tqdm # Cool progress bar

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras as ks

SEED = 458
RNG = np.random.default_rng(SEED) # Random number generator

# Joining paths
path_to_store_training_history = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '_training_history.png')
path_to_store_model = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '.keras')

# Function definition
def plot_training_history(training_history_object, list_of_metrics=None):
    """
    Input:
        training_history_object:: Object returned by model.fit() function in keras
        list_of_metrics        :: A list of metrics to be plotted. Use if you only 
                                  want to plot a subset of the total set of metrics 
                                  in the training history object. By Default it will 
                                  plot all of them in individual subplots.
    """
    history_dict = training_history_object.history
    if list_of_metrics is None:
        list_of_metrics = [key for key in list(history_dict.keys()) if 'val_' not in key]
    trainHistDF = pd.DataFrame(history_dict)
    # trainHistDF.head()
    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in train_keys]
    nr_plots = len(train_keys)
    fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
    for i in range(len(train_keys)):
        ax[i].plot(np.array(trainHistDF[train_keys[i]]), label='Training')
        ax[i].plot(np.array(trainHistDF[valid_keys[i]]), label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(train_keys[i])
        ax[i].grid('on')
        ax[i].legend()
    fig.tight_layout()
    return fig

# Fetching dataset
with h5py.File(PATH_TO_DATASET,'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X_train'])
    y_train = np.asarray(f['y_train'])
    X_test  = np.asarray(f['X_test'])
    print('Nr. train images: %i'%(X_train.shape[0]))
    print('Nr. test images: %i'%(X_test.shape[0]))

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocessing
X_train = X_train.astype("float32")/np.max(X_train)
X_test  = X_test.astype("float32")/np.max(X_test)

y_train = ks.utils.to_categorical(y_train, len(np.unique(y_train)))

# Implementing network
from tensorflow_addons.metrics import F1Score
f1_score = F1Score(num_classes=10, average='macro')

model = ks.Sequential(
    [
        ks.layers.Input((32,32,3)), # Image dimensions
        ks.layers.Flatten(),
        ks.layers.Dense(120, activation="relu"),
        ks.layers.Dense(84, activation="relu"),
        ks.layers.Dense(10, activation="softmax"), # Number of digits to be recognized
    ]
)

model.compile(loss="MSE", optimizer="adam", metrics=["accuracy", f1_score])

# Model training
start_time = time()
model_history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=(1/6), verbose=1)
end_time = time()

# Plotting training history
history_plot = plot_training_history(model_history)
history_plot.savefig(path_to_store_training_history)

# Storing model and training history
training_time = end_time - start_time
print('It took %.2f seconds to train the model for %i epochs'%(training_time, EPOCHS))
print('Storing model in %s'%path_to_store_model)
print('Storing training history in %s'%path_to_store_training_history)

history_plot.savefig(path_to_store_training_history)
model.save(path_to_store_model)