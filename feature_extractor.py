__author__ = 'George'
from pylab import *

feature_row_datatype = np.dtype([('is_walking', 'bool_'),
                                 ('x_std', '>f8'),
                                 ('y_std', '>f8'),
                                 ('z_std', '>f8'),
                                 ('magnitude_bar', '>f8'),
                                 ('magnitude_std', '>f8'),
                                 ('x_y_corr', '>f8'),
                                 ('x_z_corr', '>f8'),
                                 ('y_z_corr', '>f8'),
                                 ('fft1', '>f8'),
                                 ('fft2', '>f8'),
                                 ('fft3', '>f8'),
                                 ('fft4', '>f8')])


def extract_features(a: np.ndarray):
    features = np.empty((1, 1), dtype=feature_row_datatype)
    features['x_std'] = np.std(a['x'])
    features['y_std'] = np.std(a['y'])
    features['z_std'] = np.std(a['z'])
    features['magnitude_bar'] = np.mean(a['magnitude'])
    features['magnitude_std'] = np.std(a['magnitude'])
    # features['x_y_corr'] = np.corrcoef(a['x'], a['y'])
    return features
