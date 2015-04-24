import sqlite3
from data_parser import make_dataframes_from_filename, power_spectrum, fourier_transform, low_pass_filter, window
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
from matplotlib import rc
import math

# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

figs_path = '/Users/George/Documents/dissertation/Part-II-Dissertation/figs/'

phone_data_phone_flat_1hr, _ = make_dataframes_from_filename('assets/data/graphs/20150401235632-phone-George-Other.dat')
phone_data_phone_flat_1hr = \
    phone_data_phone_flat_1hr[(phone_data_phone_flat_1hr.index < 4400) & (phone_data_phone_flat_1hr.index > 10)]


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

tableau20 = tableau20[::2] + tableau20[1::2]


def plot_common():
    plt.ioff()
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)


def make_x_y_z_flat_plot(phone_data):
    plot_common()

    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    plt.ylim(-2, 11)
    plt.xlim(0, 4500)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    for y in range(-2, 11, 2):
        plt.plot(range(0, 4500), [y] * len(range(0, 4500)), "--", lw=0.5, color="black", alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    for i, a in enumerate(phone_data.columns - ['magnitude']):
        y_pos = phone_data[a].quantile(0.005)
        plt.text(4475, y_pos, a, fontsize=16, color=tableau20[i], ha='right')

        plt.plot(phone_data.index[::10], phone_data[a][::10], color=tableau20[i])


    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=16)

    plt.savefig(figs_path + 'x_y_z_flat_plot.pdf')


def make_timestamp_histogram(data):
    plot_common()
    a = pd.Series(data.index).diff()[1:].reset_index(drop=True)
    a = a[a > 0.0154]
    a = a[a < 0.0239]

    ax = plt.axes()
    ax.yaxis.grid(True) #vertical lines
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

    plt.xlabel("Difference in successive timestamps (s)", fontsize=16)
    plt.ylabel("Count (log scale)", fontsize=16)

    plt.savefig(figs_path + 'timestamp_histogram.pdf')


def make_noise_histogram(data):
    plot_common()

    a = data.magnitude.reset_index(drop=True)

    k = 150

    plt.xlabel("Measurements of magnitude (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=16)
    plt.ylabel("Count", fontsize=16)

    plt.hist(a, bins=k, color=tableau20[2])
    plt.savefig(figs_path + 'noise_histogram.pdf')


def make_noise_probability_plot(data):
    # plot_common()

    a = data.magnitude.reset_index(drop=True)[::5]
    plt.figure(figsize=(12, 12))
    sp.stats.probplot(a, dist="norm", plot=plt)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.title('')

    plt.xlabel("Quantiles", fontsize=16)
    plt.ylabel("Ordered values (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=16)

    plt.savefig(figs_path + 'noise_prob_plot.pdf')


def make_filter_frequency_response_plot():
    plot_common()
    # n = 5

    fs = 50
    cut_off = 5
    for i in range(1, 6):
        b, a = sp.signal.butter(i, cut_off/(fs/2.0), btype='low', analog=False, output='ba')
        w, h = sp.signal.freqz(b, a)

        plt.plot(0.5*fs*w/np.pi, np.abs(h), color=tableau20[i-1])

    plt.title('Butterworth filter frequency response', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Relative amplitude', fontsize=16)
    plt.margins(0, 0.1)

    # plt.plot(cut_off, 0.5*np.sqrt(2), 'ko')
    # plt.grid(which='both', axis='both')
    plt.legend(list(map(lambda x: "order = " + str(x), range(1,6))))

    plt.axvline(cut_off, ls='--')  # cutoff frequency
    plt.text(5.2, 0.9, 'Cut off frequency = 5 Hz', ha='left')

    plt.savefig(figs_path + 'butterworth_filters.pdf')
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

        snip = lambda x: x[(x.index > time) & (x.index < time+10)]

        phone_data = low_pass_filter(snip(phone_data))
        wear_data = low_pass_filter(snip(wear_data))

        phone_data.set_index(phone_data.index - time, inplace=True)
        wear_data.set_index(wear_data.index - time, inplace=True)

        # phone_data = window(phone_data)
        # wear_data = window(wear_data)

        make_2by2_graph_of_activity(phone_data, wear_data, activity)


def make_2by2_graph_of_activity(phone_data, wear_data, activity=""):
    plot_common()
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
    plt.suptitle("Activity = " + activity, fontsize=16)

    ax1.set_title('Phone magnitude', fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=12)
    ax1.plot(phone_data.index, phone_data.magnitude, color='k')

    phone_power_spec = power_spectrum(fourier_transform(phone_data.magnitude))
    phone_power_spec = phone_power_spec[phone_power_spec.index < 10]
    ax2.set_title('Fourier transform of\nphone magnitude', fontsize=12)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Power spectral density', fontsize=12)
    ax2.set_xlim([0, 6])
    ax2.plot(phone_power_spec.index, phone_power_spec[0], color=tableau20[0])

    ax3.plot(wear_data.index, wear_data.x, color=tableau20[0])
    ax3.plot(wear_data.index, wear_data.y, color=tableau20[1])
    ax3.plot(wear_data.index, wear_data.z, color=tableau20[2])

    ax3.set_title('Watch magnitude', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel("Acceleration (" + r'$\mathrm{ms}^{-2}$' + ")", fontsize=12)
    ax3.plot(wear_data.index, wear_data.magnitude, color='k')

    wear_power_spec = power_spectrum(fourier_transform(wear_data.magnitude))
    wear_power_spec = wear_power_spec[wear_power_spec.index < 10]
    ax4.set_title('Fourier transform of\nwatch magnitude', fontsize=12)
    ax4.set_xlabel('Frequency (Hz)', fontsize=12)
    ax4.set_ylabel('Power spectral density', fontsize=12)
    ax4.set_xlim([0, 6])
    ax4.plot(wear_power_spec.index, wear_power_spec[0], color=tableau20[0])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(figs_path + '2by2' + activity + '.pdf')

plt.ioff()
# make_noise_histogram(phone_data_phone_flat_1hr)
# make_x_y_z_flat_plot(phone_data_phone_flat_1hr)
# make_timestamp_histogram(phone_data_phone_flat_1hr)
# make_noise_probability_plot(phone_data_phone_flat_1hr)
# make_filter_frequency_response_plot()
make_2by2_of_all_activities()
