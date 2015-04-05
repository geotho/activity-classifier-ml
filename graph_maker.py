from data_parser import make_dataframes_from_filename
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
import math

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

plt.ioff()
# make_noise_histogram(phone_data_phone_flat_1hr)
# make_x_y_z_flat_plot(phone_data_phone_flat_1hr)
# make_timestamp_histogram(phone_data_phone_flat_1hr)
make_noise_probability_plot(phone_data_phone_flat_1hr)