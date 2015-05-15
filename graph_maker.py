import sqlite3
from data_parser import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
from matplotlib import rc
import math

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

latex_directory = '/Users/George/Documents/dissertation/Part-II-Dissertation/'
figs_path = latex_directory + 'figs/'

phone_data_phone_flat_1hr, _ = make_dataframes_from_filename('assets/data/graphs/20150401235632-phone-George-Other.dat')
phone_data_phone_flat_1hr = \
    phone_data_phone_flat_1hr[(phone_data_phone_flat_1hr.index < 4400) & (phone_data_phone_flat_1hr.index > 10)]
#
# global poof1
# global error
# global feature_importances
# global feature_importances_errors
# poof1, error, pooconfusion_matrices, label_encoder, feature_importances, feature_importances_errors = generate_results()
# feature_importances_one_v_rest = one_vs_rest()
print('done calculating')

sources = ['both', 'phone', 'wear']
long = ['DummyClassifier', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier']
shortest = ['Dummy', 'NB', 'DT', 'RF']
shorter = ['Dummy', 'Naive Bayes', 'Decision Tree', 'Random Forest']

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

tableaubright20 = tableau20[::2] + tableau20[1::2]
tableau20 = tableau20[1::2] + tableau20[::2]

colours = [(102, 194, 165), (252, 141, 98), (141, 160, 203), (231, 138, 195)]
for i in range(len(colours)):
    r, g, b = colours[i]
    colours[i] = (r / 255., g / 255., b / 255.)

greyscale = ['0.2', '0.4', '0.6', '0.8']

fourcolour =  [(166,206,227), (31,120,180), (178,223,138), (51,160,44)]
threecolor = [(102,194,165), (252,141,98), (141,160,203)]

for i in range(len(fourcolour)):
    r, g, b = fourcolour[i]
    fourcolour[i] = (r / 255., g / 255., b / 255.)

for i in range(len(threecolor)):
    r, g, b = threecolor[i]
    threecolor[i] = (r / 255., g / 255., b / 255.)

def fu(x):
    return x[:1].upper()


def abbreviate(activity_name):
    if activity_name == 'computeruse':
        return 'U'
    elif activity_name == 'climbing':
        return 'B'
    elif activity_name == 'gymcycling':
        return 'Y'
    elif activity_name == 'standing':
        return 'D'
    return fu(activity_name)


def capitalise(activity_name):
    if activity_name == 'computeruse':
        return 'Computer use'
    elif activity_name == 'gymcycling':
        return 'Gym cycling'
    return activity_name[:1].upper() + activity_name[1:]


def prepend(activity_name):
    a = lambda x: abbreviate(x) + ' = ' + capitalise(x)
    return a(activity_name)


def plot_common(size=(12, 9)):
    plt.ioff()
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=size)

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color('0.2')
    ax.spines["bottom"].set_color('0.2')

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', color='0.2')
    ax.tick_params(axis='y', color='0.2')

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)


def make_x_y_z_flat_plot(phone_data):
    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    plt.subplots(1, 3)
    plt.ylim(-2, 11)
    plt.xlim(0, 4500)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    # for y in range(-2, 11, 2):
    # plt.plot(range(0, 4500), [y] * len(range(0, 4500)), "--", lw=0.5, color="black", alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    for i, a in enumerate(phone_data.columns - ['magnitude']):
        ax = plt.subplot(311 + i)
        y_pos = phone_data[a].quantile(0.005)
        # ax.text(4475, y_pos, a, fontsize=20, color='k', ha='right')

        ax.plot(phone_data.index[::100], phone_data[a][::100], color='k')
        ax.set_title(a.upper() + ' Axis', fontsize=14)

        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=14)

    plt.tight_layout()
    plt.savefig(figs_path + 'x_y_z_flat_plot.pdf')
    plt.close()


def make_timestamp_histogram(data):
    plot_common()
    a = pd.Series(data.index).diff()[1:].reset_index(drop=True)
    a = a[a > 0.0154]
    a = a[a < 0.0239]

    ax = plt.axes()
    ax.yaxis.grid(True)  # vertical lines
    ax.set_axisbelow(True)

    counts, bins, patches = ax.hist(a, log=False, color=tableau20[2], bins=40)
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    ax.set_xticks(bins)
    plt.setp(ax.get_xticklabels()[::1], visible=False)
    plt.setp(ax.get_xticklabels()[::5], visible=True)
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # ax.set_xticks(bin_centers)
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%0.4f'))

    plt.xlim(bins.min(), bins.max())

    plt.xlabel("Difference in successive timestamps (s)", fontsize=20)
    plt.ylabel("Count", fontsize=20)

    plt.savefig(figs_path + 'timestamp_histogram.pdf')
    plt.close()


def make_noise_histogram(data):
    plot_common((12, 6))

    a = data.magnitude.reset_index(drop=True)

    k = 150

    plt.xlabel("Magnitude (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.title("Recorded magnitudes of a static, hour long recording", fontsize=20, y=1.1)
    plt.xlim(9.5, 9.9)

    plt.hist(a, bins=k, color=tableau20[2])
    plt.axvline(a.mean(), ls='--', linewidth=3, color='k')
    plt.axvline(9.807, ls='--', linewidth=3, color='k')
    plt.plot([a.mean(), 9.807], [11000] * 2, color='k', linewidth=2, ls='-')
    plt.text(x=(9.807 + a.mean()) / 2, y=11000, s='bias', fontsize=18, va='center', ha='center', backgroundcolor='1.0')
    plt.savefig(figs_path + 'noise_histogram.pdf', bbox_inches='tight')


def make_noise_probability_plot(data):
    # plot_common()

    a = data.magnitude.reset_index(drop=True)[::100]
    plt.figure(figsize=(12, 12))
    sp.stats.probplot(a, dist="norm", plot=plt)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on", labelsize=20)

    plt.xlabel("Quantiles", fontsize=20)
    plt.ylabel("Ordered values (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=20)
    plt.title("Probability plot of magnitude of a static, hour long recording", fontsize=20)

    plt.savefig(figs_path + 'noise_prob_plot.pdf', bbox_inches='tight')
    plt.close()


def make_filter_frequency_response_plot():
    plot_common()
    # n = 5

    fs = 50
    cut_off = 5
    for i in range(1, 6):
        b, a = sp.signal.butter(i, cut_off / (fs / 2.0), btype='low', analog=False, output='ba')
        w, h = sp.signal.freqz(b, a)

        plt.plot(0.5 * fs * w / np.pi, np.abs(h), color=tableaubright20[i - 1])

    plt.title('Butterworth filter frequency response', fontsize=20)
    plt.xlabel('Frequency (Hz)', fontsize=20)
    plt.ylabel('Relative amplitude', fontsize=20)
    plt.margins(0, 0.1)
    plt.ylim(0, 1)

    # plt.plot(cut_off, 0.5*np.sqrt(2), 'ko')
    # plt.grid(which='both', axis='both')
    plt.legend(list(map(lambda x: "order = " + str(x), range(1, 6))))

    plt.axvline(cut_off, ls='--', color='k', )  # cutoff frequency
    plt.text(5.2, 0.9, 'Critical frequency = 5 Hz', ha='left', fontsize=20)

    plt.savefig(figs_path + 'butterworth_filters.pdf')
    plt.close()
    # plt.show()


def make_2by2_of_all_activities():
    conn = sqlite3.connect('recording_database.db')
    c = conn.cursor()
    activities = c.execute("SELECT DISTINCT activity FROM data_sets").fetchall()
    activities = list(map(lambda x: x[0], activities))

    time = 100

    for activity in activities:
        phone_filename = c.execute('''SELECT filename FROM data_sets
                WHERE activity = ?
                AND device = 'phone'
                ORDER BY timestamp DESC''', (activity,)).fetchone()
        print(phone_filename[0])

        phone_data, wear_data = make_dataframes_from_filename(phone_filename[0])

        snip = lambda x: x[(x.index > time) & (x.index < time + 10)]

        phone_data = low_pass_filter(snip(phone_data))
        wear_data = low_pass_filter(snip(wear_data))

        phone_data.set_index(phone_data.index - time, inplace=True)
        wear_data.set_index(wear_data.index - time, inplace=True)

        # phone_data = window(phone_data)
        # wear_data = window(wear_data)

        make_2by2_graph_of_activity(phone_data, wear_data, activity)


def make_2by2_graph_of_activity(phone_data, wear_data, activity=""):
    plot_common()

    if activity == 'gymcycling':
        make_cover_image(phone_data)


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    ax1.plot(phone_data.index, phone_data.x, color=tableau20[0])
    ax1.plot(phone_data.index, phone_data.y, color=tableau20[1])
    ax1.plot(phone_data.index, phone_data.z, color=tableau20[2])
    plt.suptitle("Activity: " + capitalise(activity).lower(), fontsize=20)

    ax1.set_title('Phone magnitude', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=14)
    ax1.plot(phone_data.index, phone_data.magnitude, color='k')

    phone_power_spec = power_spectrum(fourier_transform(phone_data.magnitude))
    phone_power_spec = phone_power_spec[phone_power_spec.index < 10]
    ax2.set_title('Power spectrum of\nphone magnitude', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)', fontsize=14)
    ax2.set_ylabel('Power', fontsize=14)
    ax2.set_xlim([0, 6])
    ax2.plot(phone_power_spec.index, phone_power_spec[0], color='k')

    ax3.plot(wear_data.index, wear_data.x, color=tableau20[0])
    ax3.plot(wear_data.index, wear_data.y, color=tableau20[1])
    ax3.plot(wear_data.index, wear_data.z, color=tableau20[2])

    ax3.set_title('Watch magnitude', fontsize=14)
    ax3.set_xlabel('Time (s)', fontsize=14)
    ax3.set_ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=14)
    ax3.plot(wear_data.index, wear_data.magnitude, color='k')

    wear_power_spec = power_spectrum(fourier_transform(wear_data.magnitude))
    wear_power_spec = wear_power_spec[wear_power_spec.index < 10]
    ax4.set_title('Power spectrum of\nwatch magnitude', fontsize=14)
    ax4.set_xlabel('Frequency (Hz)', fontsize=14)
    ax4.set_ylabel('Power', fontsize=14)
    ax4.set_xlim([0, 6])
    ax4.plot(wear_power_spec.index, wear_power_spec[0], color='k')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # plt.savefig(figs_path + '2by2' + activity + '.pdf')
    plt.close()

def make_cover_image(phone_data):
    plot_common()
    ax1 = plt.subplot(111)
    ax1.plot(phone_data.index, phone_data.x, color=tableaubright20[0], linewidth=3)
    ax1.plot(phone_data.index, phone_data.y, color=tableaubright20[1], linewidth=3)
    ax1.plot(phone_data.index, phone_data.z, color=tableaubright20[2], linewidth=3)

    # ax1.set_title('Phone magnitude', fontsize=14)
    # ax1.set_xlabel('Time', fontsize=14)
    # ax1.set_ylabel("Acceleration", fontsize=14)

    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.plot(phone_data.index, phone_data.magnitude, color='k')
    plt.tight_layout()
    plt.savefig(figs_path + 'coverimage.pdf')
    plt.close()

def make_plot_of_data(classifier_name, f1):
    plot_common()
    rows, = f1['both'][classifier_name].shape
    m = {'both': r'Phone and watch features', 'phone': r'Phone-only features', 'wear': r'Watch-only features'}
    ind = np.arange(rows) * 2
    width = 0.35
    ax = plt.subplot(111)
    for i, s in enumerate(sources):
        ax.bar(ind + width * i, f1[s.lower()][classifier_name], width, color=threecolor[i],
               yerr=error[s.lower()][classifier_name],
               ecolor='0.2',
               label=m[s])

    ax.set_ylabel(r'$\mathrm{F}_1$ measure', size=20)

    ax.set_xticklabels(list(map(capitalise, f1['both'][classifier_name].index)), rotation=45, ha='right', size=20)
    ax.set_xticks(ind + 2.5 * width)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=9, ncol=len(sources), bbox_to_anchor=(0.5, 1.1))

    ax.set_xlim(-2 * width, np.max(ind) + 4 * width)

    ax.set_title(r'$\mathrm{F}_1$ measures per activity for all three feature sets using the random forest classifier',
                 size=20,
                 y=1.1)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off')  # labels along the bottom edge are off

    plt.savefig(figs_path + 'F1Graph_{}.pdf'.format(classifier_name),
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()
    output_f1_to_file(f1)


def make_plots_average_f1_per_featureset(f1_original):
    plot_common((12, 7))

    index_translator = lambda x: dict(zip(long, shorter))[x]
    m = {'both': r'Phone and watch features', 'phone': r'Phone-only features', 'wear': r'Watch-only features'}

    f1 = f1_original.copy()
    for k, df in f1.items():
        rows, _ = df.shape
        f1[k] = pd.DataFrame([df.mean(), df.std() / np.sqrt(rows)], index=['mean', 'std']).T
        f1[k] = f1[k].loc[long]

    rows = 4
    ind = np.arange(rows) * 2
    width = 0.35
    ax = plt.subplot(111)

    for i, c in enumerate(sources[::-1]):
        ax.barh(ind + width * i, f1[c]['mean'], width, xerr=f1[c]['std'], color=threecolor[::-1][i], ecolor='0.2',
                label=m[c])

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[::-1], labels[::-1], loc=9, ncol=len(sources), bbox_to_anchor=(0.5, 1.1))

    ax.set_yticklabels(list(map(index_translator, f1['both'].index)), size=20)
    ax.set_yticks(ind + width * 1.5)
    ax.set_xlabel(r'$\mathrm{F}_1$ measure', size=20)

    ax.set_title(r'$\mathrm{F}_1$ measures averaged across all activities',
                 size=20,
                 y=1.1)

    plt.savefig(figs_path + 'F1Graph_average_f1_per_featureset.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()


def make_one_vs_rest_plot():
    m = {'both': r'Phone and watch features', 'phone': r'Phone-only features', 'wear': r'Watch-only features'}
    plot_common()
    f1, error = one_vs_rest()
    ind = np.arange(len(f1)) * 2
    width = 0.27
    ax = plt.subplot(111)

    for i, c in enumerate(f1):
        heights = f1[c]
        err = error[c]
        ax.bar(ind + width * i, heights, width, color=colours[i], yerr=err, ecolor='0.2', label=m[c])

    ax.set_xticklabels(list(map(capitalise, f1.index)), rotation=45, ha='right', size=20)
    lgd = ax.legend(shorter, loc=9, ncol=len(shorter), bbox_to_anchor=(0.5, 1.1))

    ax.set_xticks(ind + 3 * width)
    ax.set_ylabel(r'$\mathrm{F}_1$ measure', size=20)

    ax.set_xlim(-2 * width, np.max(ind) + 4 * width)

    ax.set_title(r'$\mathrm{F}_1$ measures per activity for all three feature sets using the random forest classifier',
                 size=20,
                 y=1.1)

    plt.savefig(figs_path + 'F1GraphOneVsAll_{}.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()


def make_plot_f1_score_per_activity_for_each_classifier(device, f1):
    plot_common()
    rows, = f1[device][long[0]].shape
    ind = np.arange(rows) * 2
    width = 0.27
    ax = plt.subplot(111)

    for i, c in enumerate(long):
        height = f1[device][c]
        err = error[device][c]
        ax.bar(ind + width * i, height, width, color=fourcolour[i], yerr=err, ecolor='0.2')

    ax.set_xticklabels(list(map(capitalise, f1[device][long[0]].index)), rotation=45, ha='right', size=20)
    lgd = ax.legend(shorter, loc=9, ncol=len(shorter), bbox_to_anchor=(0.5, 1.1))
    ax.set_xticks(ind + 3 * width)
    ax.set_ylabel(r'$\mathrm{F}_1$ measure', size=20)

    ax.set_xlim(-2 * width, np.max(ind) + 4 * width)

    m = {'both': r'phone and watch', 'phone': r'phone-only', 'wear': r'watch-only'}
    s = m[device]
    ax.set_title(r'$\mathrm{F}_1$ measures per activity in ' + s + ' classification across all activities',
                 size=20,
                 y=1.1)

    plt.savefig(figs_path + 'F1GraphForEachClassifier_{}.pdf'.format(device),
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()


def output_confusion_matrices_to_file(confusion_matrices):
    for k1, v1 in confusion_matrices.items():
        for k2, v2 in v1.items():
            index = pd.Series((map(prepend, label_encoder.inverse_transform(v2.index.values))))
            v2.columns = list(map(abbreviate, label_encoder.inverse_transform(v2.columns)))
            v2.set_index(index, inplace=True)
            f = open(latex_directory + '/data/' + 'ConfusionMatrix_{}_{}.tex'.format(k1, k2), 'w')
            v2.columns.name = r'Classified as $\rightarrow$'
            # v2 = v2.div(v2.sum(axis=1), axis=0).replace(to_replace=0.0, value=np.nan)
            v2.to_latex(buf=f, escape=False)


def output_f1_to_file(f1):
    new_columns = {'DummyClassifier': 'Dummy',
                   'GaussianNB': 'NB',
                   'DecisionTreeClassifier': 'DT',
                   'RandomForestClassifier': 'RF'}
    for k, v_original in f1.items():
        v = v_original.copy()
        f = open(latex_directory + '/data/' + 'F1Table' + k + '.tex', 'w')
        v.columns = [new_columns[x] for x in v.columns]
        v = v.append(pd.DataFrame(dict(zip(v.columns, [v[x].mean() for x in v.columns])), index=['Average']))
        dd = v['Dummy']
        v.drop('Dummy', axis=1, inplace=True)
        gnb = v['NB']
        v.drop('NB', axis=1, inplace=True)
        v.insert(0, 'NB', gnb)
        v.insert(0, 'Dummy', dd)
        v.to_latex(buf=f, float_format=lambda x: '%0.3f' % x)
        make_f1_graph_average_f1_per_classifier(k, v)

    f1_copy = f1.copy()


def make_f1_graph_average_f1_per_classifier(test, v_original):
    v = v_original.copy()
    plot_common()

    plt.figure(figsize=(12, 4))
    new_columns = {'Dummy': 'Dummy',
                   'Naive Bayes': 'NB',
                   'Decision Tree': 'DT',
                   'Random Forest': 'RF'}
    columns = dict((v, k) for k, v in new_columns.items())
    v.columns = [columns[x] for x in v.columns]

    ax = plt.subplot(111)
    ind = np.arange(4) / 2 + 0.1
    width = 0.27
    rows, _ = v.shape
    errors = v[v.index != 'Average'].std() / np.sqrt(rows)
    ax.barh(ind, v.loc['Average'], width, color=tableau20[2], linewidth=1,
            xerr=errors,
            ecolor='0.2')
    ax.set_yticklabels(v.columns, size=20)
    ax.set_yticks(ind + width / 2)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_xlabel(r'$\mathrm{F}_1$ measure', size=20)

    for n in range(len(ind)):
        i = ind[n]
        vt = v.loc['Average'][n]
        err = errors[n]
        plt.text(vt + err + 0.01, i + width / 2, r'{0:.2f} $\pm$ {1:.2f}'.format(vt, err), ha='left', size=20)

    m = {'both': r'phone and watch', 'phone': r'phone-only', 'wear': r'watch-only'}
    s = m[test]
    ax.set_title(r'$\mathrm{F}_1$ measures for ' + s + ' classification averaged across all activities',
                 size=20,
                 y=1.08,
                 multialignment='center',
                 ha='center')

    plt.savefig(figs_path + 'F1GraphAverage_{}.pdf'.format(test), bbox_inches='tight')
    plt.close()


def make_feature_importances_graph(feature_importances, feature_importances_errors):
    def to_string(bad_name, dev='both'):
        d = 'phone' if bad_name[0] == 'phone' else 'watch'
        column = capitalise(bad_name[1]) if len(bad_name) == 3 else None
        measure = capitalise(bad_name[1]) if len(bad_name) == 2 else capitalise(bad_name[2])
        measure = measure.replace("_", " ")
        if measure[0:4] == 'Corr':
            column = '{} and {}'.format(measure[5].upper(), measure[7].upper())
            measure = 'Correlation'
        elif measure == 'Flatness':
            measure = 'Spectral flatness'
        elif measure == 'Std':
            measure = 'Standard deviation'
        elif measure == 'Amax':
            measure = 'Maximum amplitude'
        if column is None or column == 'Magnitude':
            column = 'magnitude'
        else:
            if len(column) == 1:
                column += ' axis '
            else:
                column += ' axes '
        if device == 'both':
            return r"{} of {} from {}".format(measure, column, d)
        return r"{} of {}".format(measure, column)


    width = 0.8
    ind = np.arange(5)

    m = {'both': r'Phone and watch', 'phone': r'Phone-only', 'wear': r'Watch-only'}
    maximum = np.max([np.max(x) for x in feature_importances.values()])
    maximum += np.max([np.max(x) for x in feature_importances_errors.values()]) / np.sqrt(50) + 0.01
    b = 0
    plt.figure(figsize=(9, 12))
    for device in m.keys():
        importances = feature_importances[device]
        ax = plt.subplot(311 + b)
        heights = list(importances.tail(5)[0])
        errors = list(feature_importances_errors[device].tail(5)[0] / np.sqrt(50))
        ax.barh(ind, heights, width, color=threecolor[b], xerr=errors, ecolor='0.2')

        ax.set_yticklabels(list(map(lambda x: to_string(x, device), importances.index)), ha='right', fontsize=20)
        ax.set_yticks(ind + 0.5 * width)
        ax.set_title('{} classification'.format(m[device]), fontsize=20)
        ax.set_xlim(0, maximum)
        ax.set_ylim(-0.01, 5.01)
        ax.set_xticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',
            left='off',
            right='off')  # labels along the bottom edge are off

        for n in range(len(ind)):
            i = ind[n]
            vt = heights[n]
            err = errors[n]
            ax.text(vt + err + 0.005, i + width / 2, r'{0:.3f} $\pm$ {1:.3f}'.format(vt, err), ha='left', size=16,
                    verticalalignment='center')

        b += 1

    plt.savefig(figs_path + 'FeatureImportancesTogether.pdf'.format(device), bbox_inches='tight')
    plt.close()


def make_one_vs_rest_feature_importance(fi):
    plot_common()
    ind = np.arange(len(fi))
    width = 0.27
    ax = plt.subplot(111)
    ax.bar(ind, [1] * 12, width, color='0.7', label='Phone features')
    ax.bar(ind, fi[0], width, color='0.2', label='Watch features')

    ax.set_xticklabels(list(map(capitalise, list(fi.index))), rotation=45, ha='right', fontsize=20)
    ax.set_xticks(ind + width)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-width, ind.max() + 2 * width)
    ax.set_ylabel('Feature importance', size=16)
    ax.set_title('Cumulative feature importances of phone and watch features for each activity',
                 size=20,
                 y=1.1,
                 multialignment='center',
                 ha='center')
    ax.axhline(fi[0].mean(), ls='--', linewidth=2, color='k')

    lgd = ax.legend(loc=9, ncol=2, bbox_to_anchor=(0.5, 1.08))

    plt.savefig(figs_path + 'FeatureImportancesCumulative.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()


plt.ioff()
make_noise_histogram(phone_data_phone_flat_1hr)
# make_x_y_z_flat_plot(phone_data_phone_flat_1hr)
# make_timestamp_histogram(phone_data_phone_flat_1hr)
# make_noise_probability_plot(phone_data_phone_flat_1hr)
# make_filter_frequency_response_plot()
# make_2by2_of_all_activities()
# output_confusion_matrices_to_file(pooconfusion_matrices)
# for c in long:
#     make_plot_of_data(c, poof1)

# for d in sources:
#     make_plot_f1_score_per_activity_for_each_classifier(d, poof1)
# make_plots_average_f1_per_featureset(poof1)
# make_one_vs_rest_plot()
# make_feature_importances_graph(feature_importances, feature_importances_errors)
# make_one_vs_rest_feature_importance(feature_importances_one_v_rest)