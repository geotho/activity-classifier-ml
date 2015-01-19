__author__ = 'George'
from pylab import *

feature_row_datatype = np.dtype([('x_std', '>f8'),
                                 ('y_std', '>f8'),
                                 ('z_std', '>f8'),
                                 ('magnitude_bar', '>f8'),
                                 ('magnitude_std', '>f8')])


def extract_features(a: np.ndarray):
    features = np.empty((1, 1), dtype=feature_row_datatype)
    features['x_std'] = np.std(a['x'])
    features['y_std'] = np.std(a['y'])
    features['z_std'] = np.std(a['z'])
    features['magnitude_bar'] = np.mean(a['magnitude'])
    features['magnitude_std'] = np.std(a['magnitude'])
    return features
