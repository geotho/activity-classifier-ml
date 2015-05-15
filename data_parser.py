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
figs_path = '/Users/George/Documents/dissertation/Part-II-Dissertation/figs/'
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
        assert features.notnull().all().all()
        assert not np.inf in features.values
    return features


def sampling_frequency(df):
    n = len(df.index)
    return n / (df.index.max() - df.index.min())


def low_pass_filter(df, cut_off=5):
    N = 4
    fs = sampling_frequency(df)
    b, a = signal.butter(N, cut_off / (fs / 2.0), btype='low', analog=False, output='ba')
    data_f = pd.DataFrame(signal.filtfilt(b, a, df, axis=0))
    data_f.index = df.index
    data_f.columns = df.columns
    return data_f


def fourier_transform(df):
    n = len(df)
    fs = sampling_frequency(df)
    dft = pd.DataFrame(np.fft.rfft(df - df.mean()))
    dft.index = np.fft.rfftfreq(n, d=1 / fs)
    return dft


def power_spectrum(dft):
    return dft.abs().apply(lambda x: x ** 2)


def spectral_flatness(power_spec):
    return sp.stats.gmean(power_spec) / np.mean(power_spec)


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

    try:
        os.remove('recording_database.db')
    except OSError:
        pass

    conn = sqlite3.connect('recording_database.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE data_sets
        (timestamp TEXT, device TEXT, user TEXT, activity TEXT, filename TEXT)''')
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

        assert phone_data.notnull().all().all()
        assert wear_data.notnull().all().all()

        assert not np.inf in phone_data.values
        assert not np.inf in wear_data.values

        data_sets[(timestamp, 'phone', user, activity)] = phone_data
        data_sets[(timestamp, 'wear', user, activity)] = wear_data

        phone_data = low_pass_filter(phone_data)
        wear_data = low_pass_filter(wear_data)

        # plt.plot(phone_data.magnitude)
        # plt.title(phone_filename)
        # plt.show()
        #

        # if activity == 'cycling':
        # plt.scatter(phone_data.x, phone_data.y)
        # plt.plot(phone_data.y, color='r')
        #
        # plt.plot(phone_data.z, color='g')
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
        flattener = lambda x: (x[0], x[1][0], x[1][1]) if x[1].__class__ == tuple else (x[0], x[1])
        phone_features.rename(columns=renaming_function('phone'), inplace=True)
        wear_features.rename(columns=renaming_function('wear'), inplace=True)
        combined_features = pd.concat([phone_features, wear_features], axis=1)
        combined_features['activity'] = activity
        combined_features = combined_features[1:-1]  # drop the first and last rows to reduce the effect of fumbling

        if data_set is None:
            data_set = combined_features
        else:
            data_set = data_set.append(combined_features)

    lb = preprocessing.LabelBinarizer()
    lb.fit(data_set['activity'])

    binary_labels = lb.transform(data_set['activity'])

    le = preprocessing.LabelEncoder()
    le.fit(data_set['activity'])
    data_set['activity'] = le.transform(data_set['activity'])

    data_set.reset_index(inplace=True)
    labels = data_set['activity']
    data_set.drop('activity', axis=1, inplace=True)
    data_set.drop('index', axis=1, inplace=True)

    return data_set, labels, binary_labels, phone_features.columns, wear_features.columns, le, lb


def generate_f1(data_set, labels, phone_columns, wear_columns, classifiers, onevsall=False):
    f1 = {}
    error = {}
    confusion = {}
    feature_importances = {}
    feature_importances_errors = {}
    sss = StratifiedShuffleSplit(labels, 10, test_size=0.5)
    for s in ['both', 'phone', 'wear']:
        f1[s] = {}
        error[s] = {}
        confusion[s] = {}

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
            # scores = f1[c.__name__]
            # confusion_for_classifier = confusion[c.__name__]

            for name, tr, te in [('both', train, test), ('phone', train_phone, test_phone),
                                 ('wear', train_wear, test_wear)]:
                if c.__name__ == 'RandomForestClassifier':
                    clf = c(n_estimators=50)
                else:
                    clf = c()
                clf.fit(tr, train_labels)
                results = pd.DataFrame(clf.predict(te))

                feature_names = phone_columns
                if name == 'both':
                    feature_names |= wear_columns
                elif name == 'wear':
                    feature_names = wear_columns

                flattener = lambda x: (x[0], x[1][0], x[1][1]) if x[1].__class__ == tuple else (x[0], x[1])
                feature_names = list(map(flattener, feature_names))

                if c.__name__ in f1[name]:
                    if c.__name__ == 'DecisionTreeClassifier':
                        tree.export_graphviz(clf,
                                             out_file=latex_directory + 'figs/{}tree.dot'.format(name),
                                             feature_names=feature_names)
                    elif c.__name__ == 'RandomForestClassifier':
                        feature_importances[name] = \
                            pd.DataFrame(clf.feature_importances_, index=feature_names).sort(0, 0, ascending=True)
                        fucking_errors = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
                        feature_importances_errors[name] = pd.DataFrame(fucking_errors, index=feature_names)
                    f1[name][c.__name__] = np.vstack(
                        (f1[name][c.__name__], f1_score(test_labels, results, average=None)))
                    confusion[name][c.__name__] += \
                        pd.DataFrame(metrics.confusion_matrix(test_labels.values, results.values))
                else:
                    f1[name][c.__name__] = f1_score(test_labels, results, average=None)
                    confusion[name][c.__name__] = \
                        pd.DataFrame(metrics.confusion_matrix(test_labels.values, results.values))

                    # print(dict(zip(list(train.columns), list(clf.feature_importances_))))

    for k, v in f1.items():
        for k1, v1 in f1[k].items():
            f1[k][k1] = np.mean(v1, axis=0)
            error[k][k1] = np.std(v1, axis=0) / np.sqrt(10)
        f1[k] = pd.DataFrame.from_dict(f1[k])
        error[k] = pd.DataFrame.from_dict(error[k])
        if not onevsall:
            f1[k].index = label_encoder.inverse_transform(f1[k].index)
            error[k].index = label_encoder.inverse_transform(error[k].index)

    return f1, error, confusion, feature_importances, feature_importances_errors


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


def one_vs_rest():
    global binary_labels
    global label_binariser
    data_set, labels, binary_labels, phone_columns, wear_columns, label_encoder, label_binariser = data_set_from_files()
    classifiers = [ensemble.RandomForestClassifier]
    sss = StratifiedShuffleSplit(labels, 10, test_size=0.5)
    global f1s
    global errors
    f1s = {}
    errors = {}
    global feature_importances
    global feature_importances_errors
    feature_importances = {}
    feature_importances_errors = {}

    binary_labels = pd.DataFrame(binary_labels, columns=label_binariser.classes_)

    for activity in binary_labels:
        f1s[activity], errors[activity], _, feature_importances[activity], feature_importances_errors[activity] = \
            generate_f1(data_set,
                        binary_labels[activity],
                        phone_columns,
                        wear_columns,
                        classifiers,
                        onevsall=True)

    cols = ['both', 'phone', 'wear']
    activities = label_binariser.classes_
    f1s = pd.DataFrame([[f1s[y][x].loc[1]['RandomForestClassifier'] for x in cols] for y in activities],
                       columns=cols,
                       index=activities)
    errors = pd.DataFrame([[errors[y][x].loc[1]['RandomForestClassifier'] for x in cols] for y in activities],
                          columns=cols,
                          index=activities)

    for k, v in feature_importances.items():
        feature_importances[k] = v['both']

    for k, v in feature_importances.items():
        v['device'] = pd.DataFrame(v.index, index=v.index).applymap(lambda x: x[0])
        feature_importances[k] = v[v.device == 'wear'][0].sum()

    # for k,v in feature_importances_errors.items():
    # feature_importances[k] = v['both']


    df = pd.DataFrame(list(feature_importances.values()), index=list(feature_importances.keys())) \
        .sort(0, ascending=False)
    return df


def generate_results():
    global label_encoder
    data_set, labels, binary_labels, phone_columns, wear_columns, label_encoder, _ = data_set_from_files()
    classifiers = [dummy.DummyClassifier,
                   naive_bayes.GaussianNB,
                   tree.DecisionTreeClassifier,
                   ensemble.RandomForestClassifier,
    ]
    f1, error, confusion, feature_importances, feature_importances_errors = \
        generate_f1(data_set, labels, phone_columns, wear_columns, classifiers)
    return f1, error, confusion, label_encoder, feature_importances, feature_importances_errors


def main():
    plt.ioff()

    # global label_encoder
    data_set, labels, binary_labels, phone_columns, wear_columns, label_encoder, _ = data_set_from_files()

    # output_data_set_metadata(labels)
    #
    # classifiers = [dummy.DummyClassifier,
    # naive_bayes.GaussianNB,
    #                tree.DecisionTreeClassifier,
    #                ensemble.RandomForestClassifier,
    # ]
    #
    # global f1
    # global confusion_matrices
    # f1, confusion_matrices = generate_f1(data_set, labels, phone_columns, wear_columns, classifiers)


    # print("Improvement of RandomForestClassifier over DummyClassifier")
    # print(f1['RandomForestClassifier'] - f1['DummyClassifier'])


if __name__ == "__main__":
    # one_vs_rest()
    main()
    # print('done')