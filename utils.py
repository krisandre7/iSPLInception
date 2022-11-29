# title       :utils
# description :Script that contains utilities that are required by or shared with other scripts
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Call it in main.py.
# notes       :Majority imports are done in here. Processing of the dataset, Model training
#              and evaluation are also done within this script.
import ast
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from datareader import DataReader
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}. Let's avoid too many logs

# **************** Default Setting ******************
# Setting default parameters for plots and figures
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Dejavu Sans'
matplotlib.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# **************** Dataset Configurations ******************
if dataset == 'daphnet':
    n_signals = 9  # Ankle acc - x,y,z; Upper Leg acc - x,y,z; Trunk acc - x,y,z;
    win_size = 192  # 1 sec -> 64 | 2.56 (3) sec -> 192 | 5 sec -> 320 # sampling rate = 64hz
    n_classes = 2  # 1 - no freeze (walk, stand, turn); 2 - freeze
    n_steps = 3  # Since we are using 3 seconds and 192 is divisible by 3
    length = 64  # Split each window of 192 time steps into sub sequences for the cnn

elif dataset == 'ispl':
    n_signals = 9  # acc - x,y,z; gyro - x,y,z; lacc - x,y,z;
    win_size = 128  # 2.56 sec -> 128 | 5 sec -> 256
    n_classes = 3  # 1 - walking; 2 - standing; 3 - sitting; 4 - running
    n_steps = 4  # 128 is divisible by 4
    length = 32  # Split each window of 128 time steps into sub sequences for the cnn

elif dataset == 'pamap2':
    n_signals = 36  # IMU hand, chest, ankle
    win_size = 256  # 2.56 sec -> 256 | 5.12 sec -> 512 # sampling rate is 100hz
    n_classes = 11  # Removed 7 classes due to lack of sufficient training data
    n_steps = 4  # 256 is divisible by 4
    length = 64  # Split each window of 256 time steps into sub sequences for the cnn

elif dataset == 'opportunity':
    n_signals = 77  # ...
    win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
    n_classes = 17
    n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
    length = 30  # Split each window of 90 time steps into sub sequences for the cnn

elif dataset == 'ucihar':
    n_signals = 9  # acc - x,y,z; gyro - x,y,z; bacc - x,y,z;
    win_size = 128  # 2.56 seconds
    n_classes = 6  # 1 - Walking; 2 - Walking_Upstairs; 3 - Walking_Downstairs; 4 - Sitting; 5 - Standing; 6 - Laying
    n_steps = 4  # 128 is divisible by 4
    length = 32  # Split each window of 128 time steps into sub sequences for the cnn


def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        start += (size / 2)


def segment(x, y, window_size, dataset_signals=9):
    #     print(f"X: {x.shape}\nY: {y.shape}\nWin_size: {window_size}\nSignals: {dataset_signals}")
    segments = np.zeros(((len(x) // (window_size // 2)) - 1, window_size, dataset_signals))
    labels = np.zeros(((len(y) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windows(x, window_size):
        if len(x[start:end]) == window_size:
            m = stats.mode(y[start:end], keepdims=False)
            segments[i_segment] = x[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


# Function that loads a specified dataset
# dataset _type is either 'original' or train_test_split
def load_dataset(dataset='ucihar', datapath='dataset/ucihar', _type='original'):
    datapath = datapath.rstrip("/")
    # Check if our desired dataset has not yet been generated
    if not os.path.exists(f'{datapath}/{dataset}.h5'):
        DataReader(dataset, datapath)

    # class labels
    with open(f'{datapath}/{dataset}.h5.classes.json', 'r') as f:
        labels = ast.literal_eval(f.read())

    with h5py.File(f'{datapath}/{dataset}.h5', 'r') as f:
        X_train, y_train = np.array(f['train']['inputs']), np.array(f['train']['targets'])
        X_val, y_val = np.array(f['validation']['inputs']), np.array(f['validation']['targets'])
        X_test, y_test = np.array(f['test']['inputs']), np.array(f['test']['targets'])

    return X_train, y_train, X_val, y_val, X_test, y_test, labels


def transform_y(y, nr_classes):
    # Transforms y, a list with one sequence of A timesteps
    # and B unique classes into a binary Numpy matrix of
    # shape (A, B)
    if dataset == "ucihar" or dataset == "ispl":
        y = np.array([a - 1 for a in y])
    ybinary = to_categorical(y, nr_classes)
    return ybinary


# Model and dataset evaluation
def evaluate_model(_model, _X_train, _y_train, _X_test, _y_test, _epochs=20, patience=10,
                   batch_size=64, _save_name='models/please_provide_a_name.h5', _log_dir='logs/fit'):
    """
    Returns the best trained model and history objects of the currently provided train & test set
    """
    early_stopping_monitor = EarlyStopping(patience=patience)

    checkpoint_path = _save_name
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = ModelCheckpoint(checkpoint_path,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  save_weights_only=False,
                                  verbose=0)
    # Tensorboard Callback
    tensorboard_callback = TensorBoard(log_dir=_log_dir, histogram_freq=1)

    # Reduce Learning rate after plateau
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10,
                                  min_lr=0.0001, verbose=1)

    # Training the model
    history = _model.fit(_X_train,
                         _y_train,
                         batch_size=batch_size,
                         validation_data=(_X_test, _y_test),
                         epochs=_epochs,
                         verbose=1,
                         # shuffle=True,
                         use_multiprocessing=True,
                         callbacks=[cp_callback, tensorboard_callback, early_stopping_monitor, reduce_lr])
    best_model = load_model(checkpoint_path)
    return best_model, history


# ................................................................................................ #
# **************** 1. Load Dataset ******************
# These have preconfigured train/test splits

X_train, y_train_int, X_val, y_val_int, X_test, y_test_int, labels = load_dataset(dataset, datapath)

if dataset != 'ucihar':
    X_train, y_train_int = segment(X_train, y_train_int, win_size, dataset_signals=n_signals)
    X_val, y_val_int = segment(X_val, y_val_int, win_size, dataset_signals=n_signals)
    X_test, y_test_int = segment(X_test, y_test_int, win_size, dataset_signals=n_signals)

y_train = transform_y(y_train_int, n_classes)
y_val = transform_y(y_val_int, n_classes)
y_test = transform_y(y_test_int, n_classes)

# Using original train/val/test split from datareader.py
# ................................................................................................ #

frequencies = y_train_int.mean(axis=0) * 100
frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
frequencies_df.plot(kind='bar', legend=False)
plt.show()


# ................................................................................................ #
# Plotting useful results from training
def plot_classification_report(cr, title='Classification Report ', with_avg_total=False,
                               cmap=plt.get_cmap('plasma'), path="images/cr.png"):
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 4)]:
        #         print(line)
        t = line.split()
        if len(t) <= 0:
            continue
        #         print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        #         print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[-2].split()[2:-1]
        #         print(aveTotal)
        classes.append('avg/total')
        vAveTotal = [float(x) for x in aveTotal]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.savefig(path)
    plt.show()


def plot_metrics(history, model, dataset, image_path):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'accuracy':
            plt.ylim([0.3, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.title(f"{name} of {model} on the {dataset} dataset")

    plt.savefig(image_path, bbox_inches='tight')
    plt.show()
