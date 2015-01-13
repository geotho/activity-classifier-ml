from pylab import *

__author__ = 'George'


def make_array_from_file(filename):
    # TODO: timestamps in numpy array
    datatype = np.dtype([('timestamp', '>i8'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    data = np.fromfile(open(filename, "rb"), datatype)
    return data


def generate_additional_columns(old_array):
    new_dt = np.dtype(old_array.dtype.descr + [('magnitude', '>f4')])
    new_array = np.zeros(old_array.shape, dtype=new_dt)
    # TODO: find a better way of copying array into another
    for label in ['timestamp', 'x', 'y', 'z']:
        new_array[label] = old_array[label]
    new_array['magnitude'] = np.sqrt(old_array['x'] ** 2 + old_array['y'] ** 2 + old_array['z'] ** 2)
    return new_array



data = generate_additional_columns(make_array_from_file("assets/data/accelData20150104.dat"))
