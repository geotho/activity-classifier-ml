from urllib.response import addinfourl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import os
import math
import sqlite3
from sklearn.metrics.metrics import f1_score
from sklearn import preprocessing, metrics
from sklearn import tree
from sklearn import dummy
from sklearn import naive_bayes
from sklearn import ensemble
from pandas.tools.plotting import autocorrelation_plot
from scipy import signal
from sklearn.cross_validation import StratifiedShuffleSplit
import sys

__author__ = 'George'

file_datatype = np.dtype([('timestamp', '>i8'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
processed_datatype = np.dtype([('timestamp', '<f12'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('magnitude', '<f4')])
latex_directory = '/Users/George/Documents/dissertation/Part-II-Dissertation/'
label_encoder = None


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


def make_dataframes_from_filename(phone_filename):
    wear_filename = phone_filename.replace("phone", "wear")

    phone_data = pd.DataFrame(generate_additional_columns(make_array_from_file(phone_filename)))
    wear_data = pd.DataFrame(generate_additional_columns(make_array_from_file(wear_filename)))

    phone_data.set_index('timestamp', inplace=True)
    wear_data.set_index('timestamp', inplace=True)

    return phone_data, wear_data


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

    def mean_average_deviation(data):
        return np.mean(np.abs(data - np.mean(data)))

    def flatness(group):
        return spectral_flatness(power_spectrum(fourier_transform(group['magnitude'])))[0]

    def entropy(group):
        return spectral_entropy(power_spectrum(fourier_transform(group['magnitude'])))[0]

    def peak_frequency(group):
        return power_spectrum(fourier_transform(group['magnitude'])).idxmax()[0]

    features = grouped.agg([np.mean, np.std, np.max, mean_average_deviation])
    functions = [corr_x_y, corr_x_z, corr_y_z, flatness, entropy, peak_frequency]
    for f in functions:
        series = grouped.apply(f)
        series.name = f.__name__
        features = pd.concat([features, series], axis=1)
    return features


def sampling_frequency(df):
    n = len(df.index)
    return n / (df.index.max()-df.index.min())


def low_pass_filter(df, cut_off=5):
    N = 4
    fs = sampling_frequency(df)
    b, a = signal.butter(N, cut_off/(fs/2.0), btype='low', analog=False, output='ba')
    data_f = pd.DataFrame(signal.filtfilt(b, a, df, axis=0))
    data_f.index = df.index
    data_f.columns = df.columns
    return data_f


def fourier_transform(df):
    n = len(df)
    fs = sampling_frequency(df)
    dft = pd.DataFrame(np.fft.rfft(df-df.mean()))
    dft.index = np.fft.rfftfreq(n, d=1/fs)
    return dft


def power_spectrum(dft):
    return dft.abs().apply(lambda x: x**2)


def spectral_flatness(power_spec):
    return sp.stats.gmean(power_spec)/np.mean(power_spec)


def spectral_entropy(power_spec):
    normalised_power_spec = power_spec / power_spec.sum()
    return -(normalised_power_spec * normalised_power_spec.apply(np.log2)).sum()


def welch(df):
    fs = sampling_frequency(df)
    return sp.signal.welch(df, fs, nperseg=1024)


def window(df):
    rows, cols = df.shape
    windowing_function = np.hanning
    window = pd.concat([pd.Series(windowing_function(len(df)))] * cols, axis=1)
    return pd.DataFrame(window.values * df.values, columns=df.columns, index=df.index)


def simple_plot(array1, array2=None, filename="Default.pdf"):
    # plot(array['timestamp'], array['x'], 'r', array['timestamp'], array['y'], 'g', array['timestamp'], array['z'], 'b')
    if array2 is None:
        plt.plot(array1.index, array1.magnitude)
    else:
        plt.plot(array1.index, array1.magnitude, 'r', array2.index, array2.magnitude, 'b')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s2)')
    plt.title(filename)
    plt.grid(True)
    plt.show()
    # plt.savefig(filename.replace(".dat", ".pdf"))


def data_set_from_files():
    data_set = None
    # filenames = {}
    conn = sqlite3.connect('recording_database.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE data_sets
        (timestamp text, device text, user text, activity text, filename text)''')
    conn.commit()

    global data_sets
    data_sets = {}

    data_directory = "assets/data/"
    for i in os.listdir(data_directory):
        if i.endswith(".dat"):
            timestamp, device, user, activity = i[:-4].lower().split('-')
            filename = data_directory + i
            c.execute("INSERT INTO data_sets VALUES (?, ?, ?, ?, ?)",
                      (timestamp, device, user, activity, filename))
            conn.commit()

    # for (timestamp, device, user, activity), phone_filename in filenames.items():
    query = "SELECT * FROM data_sets WHERE device='phone'"
    c.execute(query)
    for (timestamp, device, user, activity, phone_filename) in c.fetchall():
        phone_data, wear_data = make_dataframes_from_filename(phone_filename)

        data_sets[(timestamp, 'phone', user, activity)] = phone_data
        data_sets[(timestamp, 'wear', user, activity)] = wear_data

        phone_data = low_pass_filter(phone_data)
        wear_data = low_pass_filter(wear_data)

        # plt.plot(phone_data.magnitude)
        # plt.title(phone_filename)
        # plt.show()
        #
        # plt.plot(low_pass_filter(phone_data.magnitude))
        # plt.title(phone_filename)
        # plt.show()

        # print(phone_filename)
        # autocorrelation_plot(phone_data.magnitude-np.mean(phone_data.magnitude))
        # plt.show()
        # if len(phone_data) < len(wear_data):
        #     plt.xcorr(np.pad(phone_data.magnitude-np.mean(phone_data.magnitude), (0, len(wear_data) - len(phone_data)), mode='constant', constant_values=(9.81, 9.81)), wear_data.magnitude-np.mean(wear_data.magnitude), maxlags=30)
        # else:
        #     plt.xcorr(np.pad(wear_data.magnitude-np.mean(wear_data.magnitude), (0, len(phone_data) - len(wear_data)), mode='constant', constant_values=(9.81, 9.81)), phone_data.magnitude-np.mean(phone_data.magnitude), maxlags=30)


        # plt.xcorr()
        # plt.acorr(wear_data.magnitude-np.mean(wear_data.magnitude), maxlags=None)
        # plt.title(wear_filename)
        # plt.show()
        # b = plt.acorr(wear_data.magnitude)
        # fs = len(phone_data.index) / max(phone_data.index)
        # plt.plot(phone_data.index * fs / len(phone_data.index), np.abs(np.fft.fft(phone_data.magnitude-np.mean(phone_data.magnitude))))
        # simple_plot(phone_data, wear_data, phone_filename)

        # fourier_transform(phone_data.magnitude).abs().plot()
        # plt.title(phone_filename)
        # plt.show()
        #
        # fourier_transform((phone_data.magnitude)).abs().plot()
        # plt.title(phone_filename + "low pass filtered")
        # plt.show()

        # print(phone_filename)
        # print(spectral_entropy(power_spectrum(fourier_transform(phone_data))))

        phone_features = extract_features(bin(phone_data))
        wear_features = extract_features(bin(wear_data))

        renaming_function = lambda d: lambda xy: (d, ) + (xy, )
        phone_features.rename(columns=renaming_function('phone'), inplace=True)
        wear_features.rename(columns=renaming_function('wear'), inplace=True)
        combined_features = pd.concat([phone_features, wear_features], axis=1)
        combined_features['activity'] = activity
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

    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)

    binary_labels = lb.transform(labels)

    return data_set, labels, binary_labels, phone_features.columns, wear_features.columns, le


def generate_f1(data_set, labels, phone_columns, wear_columns, classifiers):
    f1 = {}
    error = {}
    confusion = {}
    sss = StratifiedShuffleSplit(labels, 10, test_size=0.5)
    for c in classifiers:
        f1[c.__name__] = {}
        error[c.__name__] = {}
        confusion[c.__name__] = {}

    for train_indexes, test_indexes in sss:
        train = data_set.iloc[train_indexes]
        train_phone = train[phone_columns]
        train_wear = train[wear_columns]

        test = data_set.iloc[test_indexes]
        test_phone = test[phone_columns]
        test_wear = test[wear_columns]

        train_labels = labels.iloc[train_indexes]
        test_labels = labels.iloc[test_indexes]

        test_labels = test_labels.reset_index()
        test_labels.drop('index', axis=1, inplace=True)

        for c in classifiers:
            scores = f1[c.__name__]
            confusion_for_classifier = confusion[c.__name__]

            for name, tr, te in [('both', train, test), ('phone', train_phone, test_phone), ('wear', train_wear, test_wear)]:
                clf = c()
                clf.fit(tr, train_labels)
                results = pd.DataFrame(clf.predict(te))
                if name in scores:
                    if c.__name__ == 'DecisionTreeClassifier' and name == 'both':
                        tree.export_graphviz(clf,
                                             out_file=latex_directory + 'figs/tree.dot',
                                             feature_names=phone_columns | wear_columns)
                    scores[name] = np.vstack((scores[name], f1_score(test_labels, results, average=None)))
                    confusion_for_classifier[name] += \
                        metrics.confusion_matrix(test_labels.values, results.values, labels=label_encoder.classes_)
                else:
                    scores[name] = f1_score(test_labels, results, average=None)
                    confusion_for_classifier[name] = \
                        metrics.confusion_matrix(test_labels.values, results.values, labels=label_encoder.classes_)

            # print(dict(zip(list(train.columns), list(clf.feature_importances_))))

            # clf = c()
            # clf.fit(train_phone, train_labels)
            # results = pd.DataFrame(clf.predict(test_phone))
            # if 'phone' in scores:
            #     scores['phone'] = np.vstack((scores['phone'], f1_score(test_labels, results, average=None)))
            #     confusion_for_classifier['both'] += \
            #         metrics.confusion_matrix(test_labels, results, labels=label_encoder.classes_)
            # else:
            #     scores['phone'] = f1_score(test_labels, results, average=None)
            #
            # clf = c()
            # clf.fit(train_wear, train_labels)
            # results = pd.DataFrame(clf.predict(test_wear))
            # if 'wear' in scores:
            #     scores['wear'] = np.vstack((scores['wear'], f1_score(test_labels, results, average=None)))
            # else:
            #     scores['wear'] = f1_score(test_labels, results, average=None)

    for k, v in f1.items():
        for k1, v1 in f1[k].items():
            f1[k][k1] = np.mean(v1, axis=0)
            error[k][k1] = np.std(v1, axis=0)/np.sqrt(10)
        f1[k] = pd.DataFrame.from_dict(f1[k])
        error[k] = pd.DataFrame.from_dict(error[k])
        f1[k].index = label_encoder.inverse_transform(f1[k].index)
        error[k].index = label_encoder.inverse_transform(error[k].index)

    return f1, confusion


def output_data_set_metadata(labels):
    df = pd.DataFrame()
    vc = labels.value_counts()
    df['Counts'] = vc
    df['Proportion'] = labels.value_counts(normalize=True)
    df['Time recorded (minutes)'] = (df['Counts'] / 6).apply(round).apply(int)
    d = {'Counts': df['Counts'].sum(),
         'Proportion': 1,
         'Time recorded (minutes)': df['Time recorded (minutes)'].sum()}
    df.set_index(label_encoder.inverse_transform(df.index.values), inplace=True)
    total = pd.DataFrame(data=d, index=['Total'])
    df = df.append(total)
    df.insert(0, 'Activity', df.index)
    df['Activity'] = df['Activity'].apply(lambda x: x[:1].upper() + x[1:])
    f = open(latex_directory + 'data/TableDataSetMetadata.tex', 'w')
    df.to_latex(buf=f, float_format=lambda x: '{:.2%}'.format(x), index=False)


def main():
    plt.ioff()

    try:
        os.remove('recording_database.db')
    except OSError:
        pass

    global label_encoder
    data_set, labels, binary_labels, phone_columns, wear_columns, label_encoder = data_set_from_files()

    output_data_set_metadata(labels)

    classifiers = [tree.DecisionTreeClassifier,
                   dummy.DummyClassifier,
                   ensemble.RandomForestClassifier,
                   naive_bayes.GaussianNB,
                   ]

    global f1
    global confusion_matrices
    f1, confusion_matrices = generate_f1(data_set, labels, phone_columns, wear_columns, classifiers)

    for k, v in f1.items():
        print(k)
        print(v)
        f = open(latex_directory + '/data/' + 'Table' + k + '.tex', 'w')
        v.to_latex(buf=f, float_format=lambda x: '%0.3f' % x)

    print("Improvement of RandomForestClassifier over DummyClassifier")
    print(f1['RandomForestClassifier'] - f1['DummyClassifier'])
    # POSSIBLY IGNORE CYCLING DATA BEFORE 11/03/14 - it was collected in a rucksack.

if __name__ == "__main__":
    main()