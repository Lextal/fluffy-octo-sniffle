import numpy as np
from PIL import Image
from joblib import Parallel, delayed


def pixel_class_lambda(image_path, n_classes=19):
    result = np.zeros(n_classes)
    img = Image.open(image_path).convert("L")
    img = np.asarray(img)
    cls, counts = np.unique(img, return_counts=True)
    for i, c in enumerate(cls):
        result[c] = counts[i]
    return result


def compute_class_frequencies(file_paths, class_lambda=None, n_classes=19, n_jobs=16):
    """
    :param file_paths: iterable of input files
    :param class_lambda: function (f_name, n_classes -> ndarray)that gives class distribution over a given file,
        Must return ndarray of shape (n_classes)
    :param n_classes: number of classes
    :return: normalized ndarray with a priori probability for each class

    Example:
        from glob import glob
        file_paths = glob('/data/*.png')
        cls_freqs = compute_class_frequencies(file_paths, pixel_class_lambda, n_classes=19)
    """
    stats = Parallel(n_jobs=n_jobs)(delayed(class_lambda)(f, n_classes) for f in file_paths)
    unnormalized = np.stack(stats).sum(0)
    return unnormalized / np.sum(unnormalized)
