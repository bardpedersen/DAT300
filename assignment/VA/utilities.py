import numpy as np
import gzip
import pickle

import os

SEED = 3466
RNG = np.random.default_rng(SEED) # Random number generator
   
def download_mnist():
    """
    Downloads mnist dataset
    """
    from tensorflow.keras.datasets import mnist
    (train_X, train_y), (X_test, y_test) = mnist.load_data()

    train_X = train_X.astype("float32")/np.max(train_X) # Normalizing values between [0,1]
    X_test = X_test.astype("float32")/np.max(train_X) # Normalizing values between [0,1]

    # Splitting train_X into X_train and X_val
    train_size = 5*10**4
    
    all_indices = [i for i in range(train_X.shape[0])]
    random_indices_train = RNG.choice(all_indices, size=train_size, replace=False)
    random_indices_val = [j for j in all_indices if j not in random_indices_train]

    X_train = train_X[random_indices_train,:,:]
    y_train = train_y[random_indices_train]
    X_val   = train_X[random_indices_val,:,:]
    y_val   = train_y[random_indices_val]

    if not os.path.exists('./mnist_data/'):
        os.makedirs('./mnist_data/')

    np.savetxt('./mnist_data/X_train.txt', X_train.reshape(X_train.shape[0], -1))
    np.savetxt('./mnist_data/y_train.txt', y_train)
    np.savetxt('./mnist_data/X_val.txt',   X_val.reshape(X_val.shape[0], -1))
    np.savetxt('./mnist_data/y_val.txt',   y_val)
    np.savetxt('./mnist_data/X_test.txt',  X_test.reshape(X_test.shape[0], -1))
    np.savetxt('./mnist_data/y_test.txt',  y_test)

def load_mnist(small_train_size=1000, small_val_size=300, verbose=1):
    """
    Checks if data is downloaded, downloads if not and loads it in both cases.
    """
    if not os.path.exists('./mnist_data/'):
        download_mnist()

    X_train_2D = np.loadtxt('./mnist_data/X_train.txt')
    y_train    = np.loadtxt('./mnist_data/y_train.txt')
    X_val_2D   = np.loadtxt('./mnist_data/X_val.txt')
    y_val      = np.loadtxt('./mnist_data/y_val.txt')
    X_test_2D  = np.loadtxt('./mnist_data/X_test.txt')
    y_test     = np.loadtxt('./mnist_data/y_test.txt')

    X_train = X_train_2D.reshape(X_train_2D.shape[0], X_train_2D.shape[1]//28, 28)
    X_val   = X_val_2D.reshape(  X_val_2D.shape[0],   X_val_2D.shape[1]//28,   28)
    X_test  = X_test_2D.reshape( X_test_2D.shape[0],  X_test_2D.shape[1]//28,  28)

    small_X_train = X_train[:small_train_size]
    small_y_train = y_train[:small_train_size]
    small_X_val   = X_val[:small_val_size]
    small_y_val   = y_val[:small_val_size]

    if verbose == 1:
        print('X_train shape', X_train.shape)
        print('y_train shape', y_train.shape)
        print('X_val shape', X_val.shape)
        print('y_val shape', y_val.shape)
        print('X_test shape', X_test.shape)
        print('y_test shape', y_test.shape)
        print('small_X_train shape', small_X_train.shape)
        print('small_y_train shape', small_y_train.shape)
        print('small_X_val shape', small_X_val.shape)
        print('small_y_val shape', small_y_val.shape)

    datasets = {
        'X_train'      : X_train,
        'y_train'      : y_train,
        'X_val'        : X_val,
        'y_val'        : y_val,
        'X_test'       : X_test,
        'y_test'       : y_test,
        'small_X_train': small_X_train,
        'small_y_train': small_y_train,
        'small_X_val'  : small_X_val,
        'small_y_val'  : small_y_val,
    }

    return datasets
    