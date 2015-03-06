from pylab import *
import pandas as pd
import os
from sklearn.metrics.metrics import f1_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

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


def simple_plot(array1, array2, filename="Default"):
    # plot(array['timestamp'], array['x'], 'r', array['timestamp'], array['y'], 'g', array['timestamp'], array['z'], 'b')
    plot(array1['timestamp'], array1['magnitude'], 'r', array2['timestamp'], array2['magnitude'], 'b')
    xlabel('time (s)')
    ylabel('acceleration (m/s2)')
    title(filename)
    grid(True)
    show()

files = []
data_set = None
datasets = {}
data_directory = "assets/data/"
for i in os.listdir(data_directory):
    if i.endswith(".dat"):
        timestamp, device, user, activity = i[:-4].lower().split('-')
        if device == 'phone':
            datasets[(timestamp, device, user, activity)] = data_directory + i
            print(i)

for k, phone_filename in datasets.items():
    wear_filename = phone_filename.replace("phone", "wear")

    phone_data = pd.DataFrame(generate_additional_columns(make_array_from_file(phone_filename)))
    wear_data = pd.DataFrame(generate_additional_columns(make_array_from_file(wear_filename)))
    # simple_plot(phone_data, wear_data, phone_filename)

    phone_data.set_index('timestamp', inplace=True)
    wear_data.set_index('timestamp', inplace=True)

    phone_features = extract_features(bin(phone_data))
    wear_features = extract_features(bin(wear_data))

    renaming_function = lambda d: lambda xy: (d, ) + (xy, )
    phone_features.rename(columns=renaming_function('phone'), inplace=True)
    wear_features.rename(columns=renaming_function('wear'), inplace=True)
    combined_features = pd.concat([phone_features, wear_features], axis=1)
    # combined_features = wear_features
    combined_features['activity'] = k[3]
    combined_features = combined_features[1:-1]  # drop the first and last rows to reduce the effect of fumbling

    if data_set is None:
        data_set = combined_features
    else:
        data_set = data_set.append(combined_features)

le = preprocessing.LabelEncoder()
le.fit(data_set['activity'])
data_set['activity'] = le.transform(data_set['activity'])

data_set.reset_index(inplace=True)
labels = data_set['activity']
data_set.drop('activity', axis=1, inplace=True)
data_set.drop('index', axis=1, inplace=True)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.3)

for train_indexes, test_indexes in sss:
    train = data_set.iloc[train_indexes]
    test = data_set.iloc[test_indexes]

    train_labels = labels.iloc[train_indexes]
    test_labels = labels.iloc[test_indexes]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train, train_labels)

    results = pd.DataFrame(clf.predict(test))

    test_labels = test_labels.reset_index()
    test_labels.drop('index', axis=1, inplace=True)
    a = pd.concat([results, test_labels], axis=1, ignore_index=True)
    a.columns = ['predicted', 'actual']

    wrong = a[a.predicted != a.actual]
    wrong = pd.concat([wrong.predicted.astype(int), wrong.actual.astype(int)], axis=1).applymap(le.inverse_transform)
    print(wrong)
    fi = dict(zip(list(train.columns), list(clf.feature_importances_)))
    print(f1_score(test_labels, results, [0, 1, 2, 3, 4]))
    # print(fi)
