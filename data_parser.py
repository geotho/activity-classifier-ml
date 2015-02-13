from pylab import *
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
# from feature_extractor import extract_features

__author__ = 'George'

file_datatype = np.dtype([('timestamp', '>i8'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
processed_datatype = np.dtype([('timestamp', '<f12'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('magnitude', '<f4')])


def make_array_from_file(filename):
    data = np.fromfile(open(filename, "rb"), file_datatype)
    return data


def generate_additional_columns(old_array):
    new_array = np.empty(old_array.shape, dtype=processed_datatype)
    # TODO: find a better way of copying array into another
    new_array['timestamp'] = (old_array['timestamp'] - old_array['timestamp'][0]) / 1000000000
    for label in ['x', 'y', 'z']:
        new_array[label] = old_array[label]
    new_array['magnitude'] = np.sqrt(old_array['x'] ** 2 + old_array['y'] ** 2 + old_array['z'] ** 2)
    return new_array


def bin(df):
    filtervalues = list(range(0, math.ceil(df.index.max() / 10) * 10 + 1, 10))
    bins = pd.cut(df.index, bins=filtervalues)
    return df.groupby(bins)


def extract_features(grouped):
    def pairwise_corrcoef(x, y, group):
        col1 = group[x]
        col2 = group[y]
        return np.corrcoef(col1, col2)[0, 1]

    def corr_x_y(group):
        return pairwise_corrcoef('x', 'y', group)

    def corr_x_z(group):
        return pairwise_corrcoef('x', 'z', group)

    def corr_y_z(group):
        return pairwise_corrcoef('y', 'z', group)

    features = grouped.agg([np.mean, np.std])
    functions = [corr_x_y, corr_x_z, corr_y_z]
    for f in functions:
        series = grouped.apply(f)
        series.name = f.__name__
        features = pd.concat([features, series], axis=1)
    return features


def simple_plot(array1, array2):
    # plot(array['timestamp'], array['x'], 'r', array['timestamp'], array['y'], 'g', array['timestamp'], array['z'], 'b')
    plot(array1['timestamp'], array1['magnitude'], 'r', array2['timestamp'], array2['magnitude'], 'b')
    xlabel('time (s)')
    ylabel('acceleration (m/s2)')
    title('Running. Blue = watch, Red = phone')
    grid(True)
    show()

# filename1 = "assets/data/20150209105135-phone-George-Other.dat" #static
filename1 = "assets/data/phone-George-Running-20150130183943.dat"  # good running
# filename1 = "assets/data/phone-George-Walking-20150204175429.dat"  # bad walking
# filename2 = "assets/data/wear-George-Walking-20150204175430.dat"  # bad walking
filename2 = filename1.replace("phone", "wear")

data1 = generate_additional_columns(make_array_from_file(filename1))
data2 = generate_additional_columns(make_array_from_file(filename2))

df1 = pd.DataFrame(data1)
df1.set_index('timestamp', inplace=True)
df2 = pd.DataFrame(data2)
df2.set_index('timestamp', inplace=True)

gp1 = bin(df1)
gp2 = bin(df2)
