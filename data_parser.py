from pylab import *
from feature_extractor import extract_features

__author__ = 'George'

file_datatype = np.dtype([('timestamp', '>i8'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
processed_datatype = np.dtype([('timestamp', '>f12'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('magnitude', '>f4')])

def make_array_from_file(filename):
    data = np.fromfile(open(filename, "rb"), file_datatype)
    return data


def generate_additional_columns(old_array):
    new_array = np.empty(old_array.shape, dtype=processed_datatype)
    # TODO: find a better way of copying array into another
    new_array['timestamp'] = (old_array['timestamp'] - old_array['timestamp'][0])/1000000000
    for label in ['x', 'y', 'z']:
        new_array[label] = old_array[label]
    new_array['magnitude'] = np.sqrt(old_array['x'] ** 2 + old_array['y'] ** 2 + old_array['z'] ** 2)
    return new_array


def simple_plot(array):
    # plot(array['timestamp'], array['x'], 'r', array['timestamp'], array['y'], 'g', array['timestamp'], array['z'], 'b')
    plot(array['timestamp'], array['magnitude'])
    xlabel('time (ns)')
    xlabel('time (s)')
    ylabel('acceleration (m/s2)')
    title('Plot of data')
    grid(True)
    show()

data = generate_additional_columns(make_array_from_file("assets/data/accelData20150104.dat"))
print(data)

simple_plot(data)

print(extract_features(data))
