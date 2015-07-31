import core as pem
import numpy as np
from gwpy.timeseries import TimeSeries
import optparse
import matplotlib.pyplot as plt
import scipy.special
import time
start_time = time.time()


def parse_command_line():
    """
    parse_command_line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--channel1", help="channel 1", default=None)
    parser.add_option(
        "--channel2", help="channel 2", default=None)
    parser.add_option(
        "-s", "--start-time", dest='st', help="start time", type=int,
        default=None)
    parser.add_option(
        "-e", "--end-time", dest='et', help="end time", type=int, default=None)
    parser.add_option(
        "--segment-duration", dest='sd', help="segment duration",
        type=float, default=1)
    parser.add_option(
        "--delta-f", dest='df', help="frequency range", type=float, default=1)
    parser.add_option(
        "--fhigh", help="max frequency", type=float, default=None)
    params, args = parser.parse_args()
    return params

params = parse_command_line()

channel1 = TimeSeries.fetch(params.channel1, params.st, params.et)
channel2 = TimeSeries.fetch(params.channel2, params.st, params.et)
# channel1 = TimeSeries(np.random.normal(0, 1, 16384 * 6), sample_rate=16384)
# channel2 = TimeSeries(np.random.normal(0, 1, 16384 * 6), sample_rate=16384)
if not channel1.sample_rate.value == channel1.sample_rate.value:
    decimation_val = channel1.sample_rate.value / channel2.sample_rate.value
    channel1 = channel1.resample(channel2.sample_rate.value)
max_freq = channel1.sample_rate.value / 2.
new_dec = 1
if not params.fhigh:
    params.fhigh = channel1.sample_rate.value / 2.

while (max_freq - params.fhigh) > (params.fhigh):
    print max_freq
    new_dec += 1
    max_freq *= 0.5
if (new_dec - 1):
    channel1 = channel1.resample(channel1.sample_rate.value / new_dec)
    channel2 = channel2.resample(channel2.sample_rate.value / new_dec)

snr = pem.stamp_snr(channel1, channel2, params.sd, deltaF=params.df)
print("--- %s seconds ---" % (time.time() - start_time))
# p1 = pem.psdgram(channel1, params.sd, deltaF=params.df)
# p2 = pem.psdgram(channel2, params.sd, deltaF=params.df)

# plot = snr.plot(vmin=-5, vmax=5)
# plot.add_colorbar(label='STAMP SNR')
# plot.show()

# plot = p1.plot(vmin=p1.value.min(), vmax=p1.value.max(), norm='log')
# plot.add_colorbar(label='power')
# plot.show()

# plot = p2.plot(vmin=p2.value.min(), vmax=p2.value.max(), norm='log')
# plot.add_colorbar(label='power')
# plot.show()

# N = 2.
# sigma1 = 1.
# sigma2 = 1.
# constant = (N**(2. * N) / (2.**((2. * N) - 1) * sigma2**((4. * N) + 2.)))

# x = np.arange(0.01, 10.01, 0.01)
# z = np.arange(-100, 100.01, 0.01)

# int_vals = []

# for zi in z:
#     y = (constant * np.abs(x) * np.exp(-np.abs(x * zi / sigma1**2)) *
#          scipy.special.kv(0, N * x / sigma2**2) * x**(2 * N - 1))
#     int_vals.append(100 * np.trapz(y, x))
# int_vals = np.asarray(int_vals)
# int_vals = int_vals / np.sum(int_vals)
# pat = np.real(snr.value.reshape(snr.value.size, 1))
# n, bins, patches = plt.hist(
#     pat, bins=(z[1:] + z[:-1]) / 2)
# plt.close()
# pdf = n / np.sum(n)

# tot = np.sum(pdf)
# print 'pdf sums to : ' + str(tot)
# print 'int_vals sums to : ' + str(np.sum(int_vals))
# bins_new = (bins[1:] + bins[:-1]) / 2
# cut = (z.size - bins_new.size) / 2

# fig = plt.figure()
# plt.plot(bins_new, pdf, c='r', label='pixel distribution')
# plt.plot(z, int_vals, c='b', label='theory')
# plt.plot(z[cut:-cut], np.abs(int_vals[cut:-cut] - pdf),
#          c='g', label='residual')
# ax = plt.gca()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, fontsize=12)
# ax.set_xlim(-5, 5)
# ax.set_yscale('log')
# ax.set_ylim(1e-6, .1)
# plt.show()
# plt.close()
