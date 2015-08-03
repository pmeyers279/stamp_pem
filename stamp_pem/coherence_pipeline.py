import coherence_functions as cf
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
import numpy as np
import optparse
from glue import datafind


def parse_command_line():
    """
    parse_command_line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--channel1", help="channel 1", default=None)
    parser.add_option(
        "--list", help="file containing channel list", default=None,
        type=str)
    parser.add_option(
        "-s", "--start-time", dest='st', help="start time", type=int,
        default=None)
    parser.add_option(
        "-e", "--end-time", dest='et', help="end time", type=int, default=None)
    parser.add_option(
        "--stride", dest='stride', help="stride for ffts",
        type=float, default=1)
    parser.add_option(
        "--delta-f", dest='df', help="frequency range", type=float, default=1)
    parser.add_option(
        "--fhigh", help="max frequency", type=float, default=None)
    parser.add_option(
        "--frames", help="read from frames as opposed to nds", type=int,
        default=False)
    parser.add_option(
        "--segment-duration", dest='sd', help="segment duration for specgram",
        type=float, default=None)
    params, args = parser.parse_args()
    return params


def _read_data(channel, st, et, frames=False):
    """
    get data, either from frames or from nds2
    """

    ifo = channel.split(':')[:3]
    if frames:
        # read from frames
        connection = datafind.GWDataFindHTTPConnection()
        cache = connection.find_frame_urls(ifo[0], ifo + '_R', st, et)
        data = TimeSeries.read(cache, channel, st, et)
    else:
        data = TimeSeries.fetch(channel, st, et)

    return data


def _read_list(file):
    """
    read channel list from file
    """
    channels = []
    f = open(file, 'r')
    for line in f:
        channels.append(line.split('\n')[0])
    f.close()
    return channels

params = parse_command_line()
channels = _read_list(params.list)
darm = _read_data(params.channel1, params.st, params.et, frames=params.frames)
darm_fft = cf.fftgram(darm, params.stride, pad=True)
coh = {}
# for channel in channels:
#     data = _read_data(channel, params.st, params.et, frames=params.frames)
#     # in case DARM stores too many frequencies for this channel
#     coh[channel], N, csd12, psd1, psd2 = cf.coherence(darm_fft,
#                                                       data,
#                                                       params.stride,
#                                                       overlap=None,
#                                                       pad=True)
# theorCoh = 1. / N
coh, N, csd12, psd1, psd2 = cf.coherence_list(params.channel1, channels, params.stride, st=params.st,
                                           et = params.et, pad=True)
for channel in channels:
    data = _read_data(channel, params.st, params.et, frames=params.frames)
    specgram = cf.coherence_spectrogram(darm, data, params.stride,
                                     params.sd, overlap=None, pad=True)
    plot = specgram.plot(vmin=1e-3, vmax=1, norm='log')
    plot.add_colorbar(label='coherence')
    plot.show()

theorCoh = 1. / N

for channel in channels:
    coherence = coh[channel]
    plot = coherence.plot()
    ax = plot.gca()
    freqs = coherence.frequencies
    ax.plot(freqs.value, theorCoh*np.ones(freqs.size),'r')
    ax.set_ylim(1e-5,1.1)
    ax.set_yscale('log')
    plot.show()
