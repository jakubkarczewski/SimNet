"""Different convenience methods used during validation."""
# data science libs
import numpy as np
import matplotlib.pyplot as plt

# keras imports
from keras import backend as K


def get_model_names(one_model=False):
    """Returns the function names of all the models used in testing."""
    import models
    model_names = [fun_name for fun_name in dir(models)
                   if 'model' in fun_name]

    if one_model:
        return model_names[0]
    else:
        return model_names


def get_params_dict():
    """Returns a dict with range of values for each parameter."""
    return {
        'num_pairs': (25, 50, 75, 100, 125, 150),
        'test_size': 100000,
        'model_names': tuple(get_model_names()),
        'num_bits': (16, 32, 64, 128, 256, 512),
        'training': {
            'val_split': 0.85,
            'epochs': 20,
            'batch_size': 128
        }
    }


def euclidean_distance(vects):
    """Compute euclidean distance."""
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def margin_loss(y_true, y_pred):
    m = 1
    loss = 0.5*(1-y_true)*y_pred + 0.5*y_true*K.maximum(0.0, m - y_pred)
    return loss


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


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
