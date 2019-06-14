"""Different convenience methods used during validation."""
# data science libs
import numpy as np
import matplotlib.pyplot as plt

# keras imports
from keras import backend as K

OPTIMIZED_PARAMS = {'input_shape': [41], 'l1_shape': 96, 'l2_shape': 160,
                    'l3_shape': 384, 'l4_shape': None, 'd1_rate': 0.0,
                    'd2_rate': 0.25, 'distance': 'l1'}

BASE_PARAMS = {'input_shape': [41], 'l1_shape': 60, 'l2_shape': 80,
               'l3_shape': 100, 'l4_shape': None, 'd1_rate': 0.2,
               'd2_rate': 0.2, 'distance': 'l1'}


def get_params_dict(optimized_model=False, best_pairs=False):
    """Returns a dict with range of values for each parameter."""
    return {
        'num_pairs': (50,) if best_pairs else (25, 50, 75, 100, 125, 150),
        'test_size': 100000,
        'model_params': OPTIMIZED_PARAMS if optimized_model else BASE_PARAMS,
        'num_bits': (16, 32, 64, 128, 256, 512),
        'training': {
            'val_split': 0.85,
            'epochs': 30,
            'batch_size': 128
        }
    }


def margin_loss(y_true, y_pred):
    m = 1
    loss = 0.5*(1-y_true)*y_pred + 0.5*y_true*K.maximum(0.0, m - y_pred)
    return loss


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


def plot_history(history, paths, seed=1):
    """Saves plots of traning."""
    acc_path, loss_path = paths

    plt.figure(seed + 1)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(acc_path)

    plt.figure(seed + 2)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(loss_path)
